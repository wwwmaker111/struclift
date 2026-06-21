#!/usr/bin/env python3
"""
按 BB_Source_Alignment_Pipeline.md 构建 BB↔源码对齐

- 信号 A (DWARF): 按 MD 实现：遍历真实 dwarf 条目，is_stmt 优先，空则用全部行
- 信号 B (LLVM IR): 解析 .ll 的 !dbg 元数据，IR BB → 行号
- 交叉验证: DWARF vs IR 一致→高置信度，不一致→用 IR 或交集
- 输出: block_src_lines, alignment_matrix, alignment_confidence (W)
- 默认不写出「行级回退」样本（仅保留 tree-sitter 语句档）；需旧行为时加 --keep-line-fallback
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from elftools.elf.elffile import ELFFile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_CALL, CS_GRP_RET, CS_OP_REG, CS_OP_IMM, CS_OP_MEM

try:
    from tree_sitter_languages import get_parser
    _TS_PARSER_SRC = "tree_sitter_languages"
    _HAS_TREE_SITTER = True
except ImportError:
    try:
        import tree_sitter_c as tsc
        from tree_sitter import Parser, Language
        _TS_PARSER_SRC = "tree_sitter_c"
        _HAS_TREE_SITTER = True
    except ImportError:
        _TS_PARSER_SRC = None
        _HAS_TREE_SITTER = False

STT_FUNC = 2


def _src_cache_key(p: Path) -> str:
    """源码路径缓存键：abspath + normcase，避免 Path.resolve() 在 WSL /mnt 跟符号链接极慢。"""
    try:
        return os.path.normcase(os.path.abspath(str(p.expanduser())))
    except Exception:
        return os.path.normcase(str(p))


# 按 MD 第四步：关心的语句类型（含 labeled/attributed 以覆盖更多 C 构造）
STATEMENT_TYPES = {
    "if_statement", "while_statement", "for_statement",
    "do_statement", "switch_statement", "case_statement",
    "return_statement", "expression_statement", "declaration",
    "break_statement", "continue_statement", "goto_statement",
    "compound_statement", "labeled_statement", "attributed_statement",
}
CONTROL_TYPES = {"if_statement", "while_statement", "for_statement", "do_statement", "switch_statement"}


class _LineLookup:
    """按行号取源码，不预建大 dict"""

    def __init__(self, lines: List[str]):
        self._lines = lines

    def get(self, ln: int, default: str = "") -> str:
        if 1 <= ln <= len(self._lines):
            return self._lines[ln - 1]
        return default


from struclift.config import VocabConfig
from struclift.data.dataset_ids import stable_embedding_id
from struclift.data.structures import Instruction, BasicBlock, BinaryCFG, EdgeType, PatternType
from struclift.utils.graph_utils import extract_structural_features, find_loop_headers


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 符号表
# ═══════════════════════════════════════════════════════════════════════════════

def iter_functions(elffile: ELFFile):
    symtab = elffile.get_section_by_name(".symtab") or elffile.get_section_by_name(".dynsym")
    if symtab is None:
        return
    for sym in symtab.iter_symbols():
        try:
            st_info = sym.entry["st_info"]
            st_type = getattr(st_info, "type", None) or (st_info.get("type", st_info) if isinstance(st_info, dict) else st_info)
            if st_type != "STT_FUNC" and (not isinstance(st_type, int) or (st_type & 0xF) != STT_FUNC):
                continue
            size = sym.entry["st_size"]
            if size and size > 0:
                yield sym.name, sym.entry["st_value"], size
        except (KeyError, TypeError, AttributeError):
            continue


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DWARF — 按 MD：全部条目 + 排序，get_bb_source_lines 用范围遍历
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dwarf_line_map(elffile: ELFFile) -> List[Dict]:
    """按 MD：提取所有 dwarf 行表条目（含 is_stmt），排序。"""
    entries: List[Dict] = []

    if not elffile.has_dwarf_info():
        return entries
    try:
        dwarf = elffile.get_dwarf_info()
    except Exception:
        return entries

    for cu in dwarf.iter_CUs():
        lp = dwarf.line_program_for_CU(cu)
        if lp is None:
            continue
        try:
            inc_dirs = list(lp["include_directory"] or [])
        except (KeyError, TypeError, AttributeError):
            inc_dirs = []
        try:
            fe_list = list(lp["file_entry"] or [])
        except (KeyError, TypeError, AttributeError):
            fe_list = []
        comp_dir = ""
        cu_name = ""
        try:
            top = cu.get_top_DIE()
            if "DW_AT_comp_dir" in top.attributes:
                v = top.attributes["DW_AT_comp_dir"].value
                comp_dir = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
            if "DW_AT_name" in top.attributes:
                v = top.attributes["DW_AT_name"].value
                cu_name = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        except Exception:
            pass

        cu_file_idx2path: Dict[int, str] = {}
        cu_file_idx2path_rel: Dict[int, str] = {}
        for i, fe in enumerate(fe_list):
            idx = i + 1
            try:
                name = getattr(fe, "name", None) or (fe["name"] if hasattr(fe, "__getitem__") else "")
            except (KeyError, TypeError):
                name = ""
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            name = str(name) if name else ""
            # 某些 CU 的主文件条目只有目录、没有文件名；回退到 CU 顶层真实源文件名
            if not name and idx == 1 and cu_name:
                name = cu_name
            try:
                di = getattr(fe, "dir_index", None) or (fe["dir_index"] if hasattr(fe, "__getitem__") else 0)
            except (KeyError, TypeError):
                di = 0
            name_path = Path(name) if name else Path()
            if name and name_path.is_absolute():
                path_rel = name_path
            elif di and 1 <= di <= len(inc_dirs):
                inc = inc_dirs[di - 1]
                if isinstance(inc, bytes):
                    inc = inc.decode("utf-8", errors="replace")
                path_rel = Path(inc) / name
            else:
                path_rel = name_path
            cu_file_idx2path_rel[idx] = str(path_rel)
            path = Path(path_rel)
            if comp_dir:
                path = Path(comp_dir) / path
            cu_file_idx2path[idx] = str(path)

        for entry in lp.get_entries():
            s = entry.state
            if s is None or getattr(s, "end_sequence", False):
                continue
            addr = getattr(s, "address", 0) or 0
            line = getattr(s, "line", 0) or 0
            col = getattr(s, "column", 0) or 0
            fi = getattr(s, "file", 1) or 1
            is_stmt = getattr(s, "is_stmt", False)
            if addr and line:
                entries.append({
                    "addr": addr,
                    "file_idx": fi,
                    "line": line,
                    "col": col,
                    "is_stmt": is_stmt,
                    "file_path": cu_file_idx2path.get(fi),
                    "file_path_rel": cu_file_idx2path_rel.get(fi),
                })

    entries.sort(key=lambda x: x["addr"])
    return entries


_dwarf_addrs_cache: Optional[List[int]] = None
_dwarf_addrs_entries_id: Optional[int] = None

def _get_dwarf_addrs(dwarf_entries: List[Dict]) -> List[int]:
    global _dwarf_addrs_cache, _dwarf_addrs_entries_id
    eid = id(dwarf_entries)
    if _dwarf_addrs_cache is None or _dwarf_addrs_entries_id != eid:
        _dwarf_addrs_cache = [e["addr"] for e in dwarf_entries]
        _dwarf_addrs_entries_id = eid
    return _dwarf_addrs_cache

def get_bb_source_lines(
    bb_start: int, bb_end_exclusive: int, dwarf_entries: List[Dict]
) -> List[Dict]:
    """按 MD：区间 [bb_start, bb_end_exclusive)，优先 is_stmt，空则退回全部行。"""
    all_lines: List[Dict] = []
    stmt_lines: List[Dict] = []
    addrs = _get_dwarf_addrs(dwarf_entries)
    lo = bisect.bisect_left(addrs, bb_start)
    hi = bisect.bisect_left(addrs, bb_end_exclusive)
    for e in dwarf_entries[lo:hi]:
        all_lines.append({
            "line": e["line"],
            "col": e["col"],
            "file_idx": e["file_idx"],
            "is_stmt": e["is_stmt"],
            "file_path": e.get("file_path"),
            "file_path_rel": e.get("file_path_rel"),
        })
        if e["is_stmt"]:
            stmt_lines.append({
                "line": e["line"],
                "col": e["col"],
                "file_idx": e["file_idx"],
                "file_path": e.get("file_path"),
                "file_path_rel": e.get("file_path_rel"),
            })
    return stmt_lines if stmt_lines else all_lines


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLVM IR 解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_llvm_ir_alignment(ll_path: Path) -> Dict[str, Dict[str, Any]]:
    """解析 .ll，返回 {func_name: {bb_label: {lines: set, role: str, primary_line: int?}}}"""
    content = ll_path.read_text(encoding="utf-8", errors="replace")

    dbg_locations: Dict[int, int] = {}
    for m in re.finditer(r"!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)", content):
        dbg_locations[int(m.group(1))] = int(m.group(2))

    functions: Dict[str, Dict[str, Any]] = {}
    func_def_re = re.compile(
        r'define\s+[^@]*@(?:"([^"]+)"|([^\s(]+))\s*\([^)]*\)[^{]*\{'
    )
    for m in func_def_re.finditer(content):
        func_name = m.group(1) or m.group(2)
        if not func_name:
            continue
        start = m.end()
        depth = 1
        pos = start
        while pos < len(content) and depth > 0:
            c = content[pos]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            pos += 1
        body = content[start : pos - 1]

        bb_map: Dict[str, Any] = {}
        current_bb = "entry"
        current_lines: Set[int] = set()
        current_insts: List[str] = []

        for raw_line in body.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            bb_match = re.match(r"^([a-zA-Z0-9_.]+)\s*:", line)
            if bb_match:
                if current_bb:
                    term = current_insts[-1] if current_insts else ""
                    role = "conditional_branch" if "br i1" in term else (
                        "unconditional_branch" if "br label" in term else (
                        "return" if "ret " in term else ("switch" if "switch" in term else "other")))
                    prim = None
                    td = re.search(r"!dbg\s+!(\d+)", term)
                    if td and int(td.group(1)) in dbg_locations:
                        prim = dbg_locations[int(td.group(1))]
                    bb_map[current_bb] = {"lines": current_lines.copy(), "role": role, "primary_line": prim}
                current_bb = bb_match.group(1)
                current_lines = set()
                current_insts = []
                continue
            dm = re.search(r"!dbg\s+!(\d+)", line)
            if dm:
                did = int(dm.group(1))
                if did in dbg_locations:
                    current_lines.add(dbg_locations[did])
            current_insts.append(line)

        if current_bb:
            term = current_insts[-1] if current_insts else ""
            role = "conditional_branch" if "br i1" in term else (
                "unconditional_branch" if "br label" in term else (
                "return" if "ret " in term else ("switch" if "switch" in term else "other")))
            prim = None
            td = re.search(r"!dbg\s+!(\d+)", term)
            if td and int(td.group(1)) in dbg_locations:
                prim = dbg_locations[int(td.group(1))]
            bb_map[current_bb] = {"lines": current_lines.copy(), "role": role, "primary_line": prim}

        functions[func_name] = bb_map
    return functions


def map_machine_bb_to_ir(
    block_addr_ranges: List[Tuple[int, int]],
    dwarf_entries: List[Dict],
    get_bb_source_lines_fn,
    ir_bb_map: Dict[str, Any],
) -> Dict[int, Tuple[str, float]]:
    """机器码 BB → (ir_bb_label, jaccard)。"""
    result: Dict[int, Tuple[str, float]] = {}
    for b_id, (start, end) in enumerate(block_addr_ranges):
        lines_info = get_bb_source_lines_fn(start, end + 8, dwarf_entries)  # end+8: 覆盖末条指令
        mach_lines = {x["line"] for x in lines_info}
        if not mach_lines:
            continue
        best_label = None
        best_j = 0.0
        for ir_label, ir_info in ir_bb_map.items():
            ir_lines = ir_info.get("lines") or set()
            if not ir_lines:
                continue
            inter = len(mach_lines & ir_lines)
            union = len(mach_lines | ir_lines)
            if union > 0:
                j = inter / union
                if j > best_j:
                    best_j = j
                    best_label = ir_label
        if best_label and best_j >= 0.3:
            result[b_id] = (best_label, best_j)
    return result


def cross_validate(
    block_addr_ranges: List[Tuple[int, int]],
    dwarf_entries: List[Dict],
    get_bb_source_lines_fn,
    ir_bb_map: Optional[Dict],
    mach_to_ir: Dict[int, Tuple[str, float]],
) -> Tuple[List[List[int]], List[List[float]]]:
    """融合 DWARF + IR，输出 block_src_lines 和 confidence（每 BB 每行的置信度，与 block_src_lines 对应）。"""
    block_lines: List[List[int]] = []
    block_confidence: List[List[float]] = []

    for b_id, (start, end) in enumerate(block_addr_ranges):
        dwarf_info = get_bb_source_lines_fn(start, end + 8, dwarf_entries)
        dwarf_lines = list({x["line"] for x in dwarf_info})

        ir_lines_set: Set[int] = set()
        ir_conf = 0.0
        if ir_bb_map and b_id in mach_to_ir:
            ir_label, jaccard = mach_to_ir[b_id]
            ir_info = ir_bb_map.get(ir_label)
            if ir_info:
                ir_lines_set = ir_info.get("lines") or set()
                ir_conf = jaccard

        if dwarf_lines and ir_lines_set:
            overlap = set(dwarf_lines) & ir_lines_set
            if set(dwarf_lines) == ir_lines_set:
                lines = sorted(dwarf_lines)
                conf = min(1.0, 0.7 + ir_conf * 0.3)
                block_lines.append(lines)
                block_confidence.append([conf] * len(lines))
            elif overlap and len(overlap) >= len(dwarf_lines) * 0.5:
                lines = sorted(overlap)
                conf = len(overlap) / max(len(dwarf_lines), len(ir_lines_set))
                block_lines.append(lines)
                block_confidence.append([conf] * len(lines))
            else:
                lines = sorted(ir_lines_set)
                block_lines.append(lines)
                block_confidence.append([0.5 + ir_conf * 0.3] * len(lines))
        elif dwarf_lines:
            block_lines.append(sorted(dwarf_lines))
            block_confidence.append([0.8] * len(dwarf_lines))
        elif ir_lines_set:
            block_lines.append(sorted(ir_lines_set))
            block_confidence.append([0.6 + ir_conf * 0.2] * len(ir_lines_set))
        else:
            block_lines.append([])
            block_confidence.append([])

    return block_lines, block_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. 按 MD 第四步：tree-sitter 语句映射
# 按 MD：解析完整 source_code，按文件缓存，每文件只解析一次
_ts_parser: Optional[Any] = None
_stmt_parse_cache: Dict[str, Tuple[List[Dict], Dict[int, List[Dict]]]] = {}


def _get_parser():
    global _ts_parser
    if _ts_parser is None and _HAS_TREE_SITTER:
        if _TS_PARSER_SRC == "tree_sitter_languages":
            _ts_parser = get_parser("c")
        else:
            try:
                _ts_parser = Parser(tsc.language())
            except TypeError:
                try:
                    lang = Language(tsc.language())
                    _ts_parser = Parser(lang)
                except Exception:
                    _ts_parser = Parser()
                    _ts_parser.language = Language(tsc.language())
    return _ts_parser


def _parse_snippet_and_offset(
    snippet: str, line_offset: int
) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """解析代码片段，将语句行号加上 line_offset 转为原文件行号"""
    parser = _get_parser()
    if parser is None:
        return [], {}
    try:
        tree = parser.parse(bytes(snippet, "utf-8"))
    except Exception:
        return [], {}

    def _node_type(node) -> str:
        t = getattr(node, "type", None) or getattr(node, "symbol", str(node))
        return t.decode("utf-8") if isinstance(t, bytes) else str(t)

    statements: List[Dict] = []

    # 显式栈前序遍历：FFmpeg 等巨型 .c 的 tree-sitter AST 深度可达数千层，
    # 递归 collect() 会触发 RecursionError（默认 ~1000）。
    stack: List[Tuple[Any, int]] = [(tree.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        ntype = _node_type(node)
        if ntype in STATEMENT_TYPES:
            sline = node.start_point[0] + 1 + line_offset
            eline = node.end_point[0] + 1 + line_offset
            statements.append({
                "id": len(statements),
                "type": ntype,
                "start_line": sline,
                "end_line": eline,
                "start_col": node.start_point[1],
                "end_col": node.end_point[1],
                "depth": depth,
                "is_control": ntype in CONTROL_TYPES,
            })
        ch = getattr(node, "children", None) or []
        for c in reversed(ch):
            stack.append((c, depth + 1))

    # 大文件(>100k行)全量解析可能失败，fallback 分块解析（每块 50k 行，不再递归）
    if not statements and snippet:
        lines = snippet.splitlines()
        if len(lines) > 100000:
            chunk_size = 50000
            for i in range(0, len(lines), chunk_size):
                chunk = "\n".join(lines[i : i + chunk_size])
                st, _ = _parse_snippet_and_offset(chunk, i)
                for s in st:
                    statements.append({
                        **s, "id": len(statements),
                        "start_line": s["start_line"], "end_line": s["end_line"],
                    })
            if statements:
                line_to_stmts_merged = defaultdict(list)
                for stmt in statements:
                    for ln in range(stmt["start_line"], stmt["end_line"] + 1):
                        line_to_stmts_merged[ln].append(stmt)
                return statements, dict(line_to_stmts_merged)

    line_to_stmts: Dict[int, List[Dict]] = defaultdict(list)
    for stmt in statements:
        for ln in range(stmt["start_line"], stmt["end_line"] + 1):
            line_to_stmts[ln].append(stmt)
    return statements, dict(line_to_stmts)


def build_source_statement_map(source_code: str, cache_key: Optional[str] = None):
    """兼容旧接口：解析全文。大文件易失败，不推荐。"""
    if not _HAS_TREE_SITTER:
        return [], {}
    return _parse_snippet_and_offset(source_code, 0)


def map_lines_to_statement(
    lines: Set[int], line_to_stmts: Dict[int, List[Dict]], statements: List[Dict]
) -> Tuple[Optional[int], float]:
    """
    按 MD：将行号集合映射到最合适的源码语句，返回 (stmt_id, confidence)
    每行只保留最深（最具体）的非 compound_statement，避免外层容器污染候选集。
    """
    stmt_by_id = {s["id"]: s for s in statements}
    candidate_ids: Set[int] = set()
    for line in lines:
        stmts_at_line = line_to_stmts.get(line, [])
        if not stmts_at_line:
            continue
        non_compound = [s for s in stmts_at_line if s["type"] != "compound_statement"]
        pool = non_compound if non_compound else stmts_at_line
        best = max(pool, key=lambda s: (s["depth"], -(s["end_line"] - s["start_line"])))
        candidate_ids.add(best["id"])

    if not candidate_ids:
        return None, 0.0

    if len(candidate_ids) > 1:
        non_compound = {sid for sid in candidate_ids
                        if sid in stmt_by_id and stmt_by_id[sid]["type"] != "compound_statement"}
        if non_compound:
            candidate_ids = non_compound

    if len(candidate_ids) == 1:
        return list(candidate_ids)[0], 0.95

    candidates = [stmt_by_id[sid] for sid in candidate_ids if sid in stmt_by_id]
    control_stmts = [s for s in candidates if s["is_control"]]
    if len(control_stmts) == 1:
        return control_stmts[0]["id"], 0.85

    deepest = max(candidates, key=lambda s: s["depth"])
    same_depth = [s for s in candidates if s["depth"] == deepest["depth"]]
    narrowest = min(same_depth, key=lambda s: s["end_line"] - s["start_line"])
    confidence = 0.80 if len(same_depth) == 1 else 0.60
    return narrowest["id"], confidence


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. 按 MD 第五、六步：机器码↔IR 映射 + 最终 M_gt 生成
# ═══════════════════════════════════════════════════════════════════════════════

def map_machine_bb_to_ir_bb(
    machine_bbs: List[Dict], dwarf_entries: List[Dict],
    get_bb_source_lines_fn,
    ir_bb_map: Dict[str, Any],
    prim_file_path: Optional[str] = None,
    prim_file_rel: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """按 MD：机器码 BB 与 IR BB 对应，返回 {addr: {ir_label, confidence}}"""
    mapping: Dict[int, Dict[str, Any]] = {}
    for mach_bb in machine_bbs:
        mach_lines_info = get_bb_source_lines_fn(
            mach_bb["addr"], mach_bb["addr"] + mach_bb["size"], dwarf_entries
        )
        if prim_file_path or prim_file_rel:
            mach_lines_info = [
                x for x in mach_lines_info
                if (prim_file_path and x.get("file_path") == prim_file_path)
                or (prim_file_rel and x.get("file_path_rel") == prim_file_rel)
            ]
        mach_line_set = {x["line"] for x in mach_lines_info}
        if not mach_line_set:
            continue
        best_ir = None
        best_j = 0.0
        for ir_label, ir_info in ir_bb_map.items():
            ir_lines = ir_info.get("lines") or set()
            if not ir_lines:
                continue
            overlap = len(mach_line_set & ir_lines)
            union = len(mach_line_set | ir_lines)
            if union > 0:
                j = overlap / union
                if j > best_j:
                    best_j = j
                    best_ir = ir_label
        if best_ir and best_j >= 0.3:
            mapping[mach_bb["addr"]] = {"ir_label": best_ir, "confidence": best_j}
    return mapping


def generate_alignment_ground_truth(
    machine_bbs: List[Dict],
    source_statements: List[Dict],
    dwarf_entries: List[Dict],
    ir_bb_map: Optional[Dict[str, Any]],
    line_to_stmts: Dict[int, List[Dict]],
    get_bb_source_lines_fn,
    prim_file_path: Optional[str] = None,
    prim_file_rel: Optional[str] = None,
) -> Tuple[List[List[float]], List[List[float]], Dict[str, int]]:
    """
    按 MD 第六步：生成 M_gt 和 W
    prim_file_path/prim_file_rel: 若提供则只使用该主文件的 dwarf 行（避免头文件行号污染）
    返回: (M_gt, W_mat, signal_stats)
      signal_stats: {n_both, n_agree, n_near, n_conflict, n_dwarf_only, n_ir_only, n_none}
    """
    n_b = len(machine_bbs)
    n_s = len(source_statements)
    M_gt = [[0.0] * n_s for _ in range(n_b)]
    W_mat = [[0.0] * n_s for _ in range(n_b)]

    stats = {"n_both": 0, "n_agree": 0, "n_near": 0, "n_conflict": 0,
             "n_dwarf_only": 0, "n_ir_only": 0, "n_none": 0}

    ir_bb_map = ir_bb_map or {}
    mach_to_ir = map_machine_bb_to_ir_bb(
        machine_bbs, dwarf_entries, get_bb_source_lines_fn, ir_bb_map,
        prim_file_path=prim_file_path, prim_file_rel=prim_file_rel,
    )

    for i, mach_bb in enumerate(machine_bbs):
        dwarf_info = get_bb_source_lines_fn(
            mach_bb["addr"], mach_bb["addr"] + mach_bb["size"], dwarf_entries
        )
        if prim_file_path or prim_file_rel:
            dwarf_info = [
                x for x in dwarf_info
                if (prim_file_path and x.get("file_path") == prim_file_path)
                or (prim_file_rel and x.get("file_path_rel") == prim_file_rel)
            ]
        dwarf_lines = {x["line"] for x in dwarf_info}
        dwarf_stmt_id, dwarf_conf = map_lines_to_statement(dwarf_lines, line_to_stmts, source_statements)

        ir_stmt_id: Optional[int] = None
        ir_conf = 0.0
        ir_mapping = mach_to_ir.get(mach_bb["addr"])
        if ir_mapping:
            ir_label = ir_mapping["ir_label"]
            ir_info = ir_bb_map.get(ir_label)
            if ir_info:
                ir_lines = ir_info.get("lines") or set()
                ir_stmt_id, raw_ir_conf = map_lines_to_statement(ir_lines, line_to_stmts, source_statements)
                ir_conf = raw_ir_conf * ir_mapping["confidence"] if ir_stmt_id is not None else 0.0

        if dwarf_stmt_id is not None and ir_stmt_id is not None:
            stats["n_both"] += 1
            if dwarf_stmt_id == ir_stmt_id:
                stats["n_agree"] += 1
                M_gt[i][dwarf_stmt_id] = 1.0
                W_mat[i][dwarf_stmt_id] = min(dwarf_conf + ir_conf, 1.0)
            else:
                ds = source_statements[dwarf_stmt_id]
                irs = source_statements[ir_stmt_id]
                d_range = set(range(ds["start_line"], ds["end_line"] + 1))
                i_range = set(range(irs["start_line"], irs["end_line"] + 1))
                if d_range & i_range or abs(ds["start_line"] - irs["start_line"]) <= 2:
                    stats["n_near"] += 1
                    M_gt[i][dwarf_stmt_id] = 1.0
                    W_mat[i][dwarf_stmt_id] = min(dwarf_conf + ir_conf, 1.0) * 0.85
                else:
                    stats["n_conflict"] += 1
                    M_gt[i][dwarf_stmt_id] = 0.6
                    W_mat[i][dwarf_stmt_id] = dwarf_conf * 0.5
                    M_gt[i][ir_stmt_id] = 0.4
                    W_mat[i][ir_stmt_id] = ir_conf * 0.5
        elif dwarf_stmt_id is not None:
            stats["n_dwarf_only"] += 1
            M_gt[i][dwarf_stmt_id] = 1.0
            W_mat[i][dwarf_stmt_id] = dwarf_conf * 0.80
        elif ir_stmt_id is not None:
            stats["n_ir_only"] += 1
            M_gt[i][ir_stmt_id] = 1.0
            W_mat[i][ir_stmt_id] = ir_conf * 0.80
        else:
            stats["n_none"] += 1

    for i in range(n_b):
        row_sum = sum(M_gt[i])
        if row_sum > 0:
            for j in range(n_s):
                M_gt[i][j] /= row_sum

    return M_gt, W_mat, stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Binary CFG（复用原逻辑）
# ═══════════════════════════════════════════════════════════════════════════════

def _operand_type_to_id(t: int) -> int:
    return {0: 0, CS_OP_REG: 1, CS_OP_IMM: 2, CS_OP_MEM: 3, 4: 4}.get(t, 0)


def _value_to_bucket(v: int, n: int = 64) -> int:
    if v == 0:
        return 0
    a = abs(v)
    return min(max(0, int(math.log2(a)) + 1), n - 1) if a >= 2 else 1


_capstone_md: Optional[Any] = None

def _get_capstone():
    global _capstone_md
    if _capstone_md is None:
        _capstone_md = Cs(CS_ARCH_X86, CS_MODE_64)
        _capstone_md.detail = True
    return _capstone_md

def build_cfg_for_function(
    code_bytes: bytes,
    base_addr: int,
    func_name: str,
    num_opcodes: int = 1024,
) -> Tuple[BinaryCFG, List[Tuple[int, int]]]:
    md = _get_capstone()
    insns = list(md.disasm(code_bytes, base_addr))
    if not insns:
        return BinaryCFG(func_name=func_name, blocks=[], edges=[]), []
    addr2idx = {insn.address: i for i, insn in enumerate(insns)}
    block_starts = {insns[0].address}
    for insn in insns:
        g = insn.groups
        if CS_GRP_JUMP in g or CS_GRP_RET in g:
            ni = addr2idx[insn.address] + 1
            if ni < len(insns):
                block_starts.add(insns[ni].address)
            if CS_GRP_JUMP in g and insn.operands and insn.operands[0].type == CS_OP_IMM:
                t = insn.operands[0].imm
                if t in addr2idx:
                    block_starts.add(t)
    sorted_starts = sorted(block_starts)
    addr2block: Dict[int, int] = {}
    blocks: List[BasicBlock] = []
    block_addr_ranges: List[Tuple[int, int]] = []

    for b_id, sa in enumerate(sorted_starts):
        si = addr2idx[sa]
        if b_id + 1 < len(sorted_starts):
            ei = addr2idx[sorted_starts[b_id + 1]]
        else:
            ei = len(insns)
        ea = insns[ei - 1].address + insns[ei - 1].size if ei > 0 else sa
        insts = []
        for i in range(si, ei):
            insn = insns[i]
            addr2block[insn.address] = b_id
            ot = [_operand_type_to_id(op.type) for op in insn.operands[:3]] + [0, 0, 0]
            vb = [_value_to_bucket(op.imm) if op.type == CS_OP_IMM else (op.reg & 0x3F if op.type == CS_OP_REG else 0) for op in insn.operands[:3]] + [0, 0, 0]
            insts.append(Instruction(
                opcode=stable_embedding_id(str(insn.mnemonic), num_opcodes),
                operand_types=ot[:3],
                value_buckets=vb[:3],
                raw_text=f"{insn.mnemonic} {insn.op_str}",
            ))
        blocks.append(BasicBlock(block_id=b_id, instructions=insts))
        block_addr_ranges.append((sa, ea))

    edges: List[Tuple[int, int, int]] = []
    for idx, insn in enumerate(insns):
        b_id = addr2block[insn.address]
        ni = idx + 1
        is_last = (ni >= len(insns)) or (addr2block.get(insns[ni].address, -1) != b_id)
        if not is_last:
            continue
        g = insn.groups
        if CS_GRP_RET in g:
            continue
        if CS_GRP_CALL in g:
            if ni < len(insns):
                edges.append((b_id, addr2block[insns[ni].address], int(EdgeType.CALL)))
            continue
        if CS_GRP_JUMP in g:
            is_uncond = insn.mnemonic in ('jmp', 'ljmp')
            if insn.operands and insn.operands[0].type == CS_OP_IMM:
                t = insn.operands[0].imm
                if t in addr2block:
                    tb = addr2block[t]
                    if tb <= b_id:
                        edges.append((b_id, tb, int(EdgeType.BACK)))
                    elif is_uncond:
                        edges.append((b_id, tb, int(EdgeType.UNCOND)))
                    else:
                        edges.append((b_id, tb, int(EdgeType.COND)))
            if not is_uncond and ni < len(insns):
                edges.append((b_id, addr2block[insns[ni].address], int(EdgeType.COND)))
            continue
        if ni < len(insns):
            edges.append((b_id, addr2block[insns[ni].address], int(EdgeType.UNCOND)))
    edges = list({(s, d, t) for s, d, t in edges})
    return BinaryCFG(func_name=func_name, blocks=blocks, edges=edges), block_addr_ranges


def assign_pattern_labels(cfg: BinaryCFG) -> List[int]:
    import torch
    n = cfg.num_blocks
    labels = [int(PatternType.SEQUENTIAL)] * n
    if n == 0 or not cfg.edges:
        return labels
    ei = torch.tensor([[e[0] for e in cfg.edges], [e[1] for e in cfg.edges]], dtype=torch.int64)
    lh = find_loop_headers(ei, n)
    for i in lh:
        labels[i] = int(PatternType.LOOP_HEADER)
    for s, d, et in cfg.edges:
        if et == int(EdgeType.BACK) and d in lh and labels[s] == int(PatternType.SEQUENTIAL):
            labels[s] = int(PatternType.LOOP_BODY)
    return labels


def build_alignment_matrix(
    block_lines: List[List[int]], line_to_stmt_idx: Dict[int, int],
    block_confidence: Optional[List[List[float]]] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    """返回 (M_gt, W)。block_confidence 来自 cross_validate，用于 MD 信号乘子。"""
    line_nums = sorted(line_to_stmt_idx.keys())
    n_stmt = len(line_nums)
    n_block = len(block_lines)
    if n_stmt == 0 or n_block == 0:
        return [], []
    M = [[0.0] * n_stmt for _ in range(n_block)]
    W = [[0.0] * n_stmt for _ in range(n_block)]
    for b_id, lines in enumerate(block_lines):
        if not lines:
            continue
        # 收集本块每行对应的置信度
        confs: List[float] = []
        if block_confidence and b_id < len(block_confidence) and len(block_confidence[b_id]) == len(lines):
            confs = block_confidence[b_id]
        else:
            confs = [0.85] * len(lines)  # 无 block_confidence 时回退到 0.85
        for li, ln in enumerate(lines):
            if ln in line_to_stmt_idx:
                s_idx = line_to_stmt_idx[ln]
                M[b_id][s_idx] = 1.0
                # 同一 cell 可能被多行映射，取最大置信度
                c = confs[li] if li < len(confs) else 0.85
                W[b_id][s_idx] = max(W[b_id][s_idx], c)
        row_sum = sum(M[b_id])
        if row_sum > 0:
            for j in range(n_stmt):
                M[b_id][j] /= row_sum
                if M[b_id][j] > 0 and W[b_id][j] > 0:
                    W[b_id][j] /= row_sum  # 归一化，与 M 一致
                elif M[b_id][j] > 0:
                    W[b_id][j] = 0.85 / row_sum  # 兜底
    return M, W


def sync_block_src_lines_with_alignment_matrix(
    block_lines: List[List[int]],
    block_confidence: Optional[List[List[float]]],
    alignment_matrix: List[List[float]],
) -> None:
    """
    若某 BB 在 alignment_matrix 上行和为 0，但 block_src_lines 仍列出行号，则审计会报 FAIL。
    原因：DWARF/IR 行号与 tree-sitter/回退列空间不一致。此处清空无矩阵支撑的块级行号，使 jsonl 自洽。
    （会丢失「未映射上的 DWARF 行」元数据，但避免训练/校验与 M_gt 矛盾。）
    """
    if not alignment_matrix or not block_lines:
        return
    n = min(len(alignment_matrix), len(block_lines))
    for i in range(n):
        row = alignment_matrix[i]
        if not row or sum(row) <= 1e-12:
            if block_lines[i]:
                block_lines[i] = []
            if block_confidence and i < len(block_confidence):
                block_confidence[i] = []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llvm-ir", default=None, help=".ll 文件路径（可选）")
    ap.add_argument("--llvm-ir-dir", default=None, help="逐文件 .ll 目录（替代 --llvm-ir，避免 llvm-link 静态函数重命名）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个样本（0=全部）")
    ap.add_argument(
        "--opt",
        default="O0",
        help="仅写入每条样本的 opt_level 元数据；对齐与 CFG 完全由 --elf（及 IR）决定，勿指望改此参数改变算法行为。",
    )
    ap.add_argument("--skip-alignment", action="store_true", help="跳过 tree-sitter 语句级对齐，仅用行级（Module A 用，显著加速）")
    ap.add_argument(
        "--keep-line-fallback",
        action="store_true",
        help="保留「行级回退」样本（无 tree-sitter 语句重叠时按行编码）；默认不写出行级档，jsonl 仅含语句对齐",
    )
    ap.add_argument("--workers", type=int, default=8, help="预解析源码文件的并行度")
    ap.add_argument("-v", "--verbose", action="store_true")
    _vc = VocabConfig()
    ap.add_argument(
        "--num-opcodes",
        type=int,
        default=_vc.num_opcodes,
        help="Opcode 词表大小（与训练 VocabConfig.num_opcodes 一致；有效 id 1..N-1，0 为 pad）",
    )
    ap.add_argument(
        "--src-vocab-size",
        type=int,
        default=32000,
        help="源码 token 词表大小（与 SourceCFGEncoder 的 src_vocab_size 一致）",
    )
    args = ap.parse_args()
    if args.num_opcodes < 2:
        ap.error("--num-opcodes 必须 >= 2（需保留 padding 索引 0）")
    if args.src_vocab_size < 2:
        ap.error("--src-vocab-size 必须 >= 2")
    print(
        f"词表: num_opcodes={args.num_opcodes}, src_vocab_size={args.src_vocab_size}",
        flush=True,
    )
    if args.skip_alignment and not args.keep_line_fallback:
        print(
            "注意: --skip-alignment 且无 --keep-line-fallback 时，几乎全部样本会因「行级档」被跳过，"
            "输出可能为空。",
            flush=True,
        )
    elif not args.skip_alignment and not args.keep_line_fallback:
        print("已启用: 跳过行级回退样本，仅写出 tree-sitter 语句对齐 (src_encoding_tier=stmt)", flush=True)

    elf_path = Path(args.elf)
    src_root = Path(args.src)
    out_path = Path(args.out)
    ll_path = Path(args.llvm_ir) if args.llvm_ir and Path(args.llvm_ir).exists() else None
    if args.llvm_ir and not ll_path and not args.skip_alignment:
        print("警告: 未找到 IR 文件，将仅用 DWARF，置信度会偏低。请提供 --llvm-ir 指向的 .ll 文件。", flush=True)

    ir_functions: Dict[str, Dict] = {}
    if ll_path and not args.skip_alignment:
        print("解析 LLVM IR ...", flush=True)
        ir_functions = parse_llvm_ir_alignment(ll_path)
        print(f"  IR 函数数: {len(ir_functions)}", flush=True)

    ir_dir: Optional[Path] = None
    ir_dir_stems: Set[str] = set()
    _ir_dir_cache: Dict[str, Dict[str, Dict]] = {}
    if args.llvm_ir_dir:
        _d = Path(args.llvm_ir_dir)
        if _d.is_dir():
            ir_dir = _d
            ir_dir_stems = {f.stem for f in ir_dir.glob("*.ll")}
            print(f"逐文件 IR 目录: {ir_dir} ({len(ir_dir_stems)} 个 .ll 文件)", flush=True)
        else:
            print(f"警告: --llvm-ir-dir {args.llvm_ir_dir} 不是目录", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with elf_path.open("rb") as f, out_path.open("w", encoding="utf-8") as out_f:
        elffile = ELFFile(f)
        text_sec = elffile.get_section_by_name(".text")
        text_bytes = text_sec.data()
        base_addr = text_sec["sh_addr"]

        if args.skip_alignment:
            print("快速模式 (--skip-alignment): 跳过 tree-sitter，仅用行级对齐", flush=True)
        print("解析 DWARF (MD 风格: 全部条目, is_stmt 优先, 空则 fallback) ...", flush=True)
        dwarf_entries = extract_dwarf_line_map(elffile)
        print(f"  dwarf 条目数: {len(dwarf_entries)}", flush=True)

        def get_lines(s: int, e: int, entries: List[Dict]) -> List[Dict]:
            return get_bb_source_lines(s, e, entries)

        comp_dir = src_root
        _src_cache: Dict[str, Tuple[List[str], Dict[int, str]]] = {}
        _ts_debug_printed = False
        written = 0
        total = 0
        n_tree_sitter = 0
        n_fallback = 0
        n_fallback_no_src = 0
        n_fallback_no_stmts = 0
        n_skipped_no_src = 0
        n_skipped_line_fallback = 0
        diag_no_src: List[Tuple[str, bool, List[int], List[Tuple[int, str, bool]]]] = []
        diag_no_stmts: List[Tuple[str, Optional[str], int, int, Optional[Tuple[int, int]]]] = []
        func_list = list(iter_functions(elffile))
        total_funcs = len(func_list)

        # 预解析所有会碰到的源文件（并行），避免主循环里逐个解析导致极慢
        if not args.skip_alignment and _HAS_TREE_SITTER and dwarf_entries:
            seen_cache_keys: Dict[str, Path] = {}
            seen_source_pairs = {
                (e.get("file_path"), e.get("file_path_rel"))
                for e in dwarf_entries
                if e.get("file_path")
            }
            for raw_path, rel in seen_source_pairs:
                p = Path(raw_path)
                if p.suffix not in (".c", ".h", ".hpp", ".cpp"):
                    continue
                for c in [p, comp_dir / p.name, comp_dir / p, (comp_dir / rel) if rel else None]:
                    if c is None or not c.exists() or not c.is_file():
                        continue
                    try:
                        ck = _src_cache_key(c)
                        if ck not in seen_cache_keys:
                            seen_cache_keys[ck] = c
                    except Exception:
                        pass
                    break
            to_parse = list(seen_cache_keys.items())
            if to_parse:
                n_workers = max(1, min(args.workers or 8, len(to_parse)))
                print(f"  预解析 {len(to_parse)} 个源文件 (并行 {n_workers}) ...", flush=True)

                def _load_and_parse(item: Tuple[str, Path]) -> Optional[Tuple[str, List[str], Any, List[Dict], Dict]]:
                    cache_key, path = item
                    try:
                        content = path.read_text(encoding="utf-8", errors="replace")
                        lines_list = content.splitlines()
                        line_to_text = _LineLookup(lines_list)
                        stmts, line_to_stmts = _parse_snippet_and_offset(content, 0)
                        return (cache_key, lines_list, line_to_text, stmts, line_to_stmts)
                    except Exception:
                        return None

                done = 0
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = {ex.submit(_load_and_parse, item): item for item in to_parse}
                    for fut in as_completed(futures):
                        result = fut.result()
                        if result:
                            ck, lines_list, line_to_text, stmts, line_to_stmts = result
                            _src_cache[ck] = (lines_list, line_to_text)
                            _stmt_parse_cache[ck] = (stmts, line_to_stmts)
                        done += 1
                        if done % 50 == 0 or done == len(to_parse):
                            print(f"    预解析 {done}/{len(to_parse)}", flush=True)
                print(f"  预解析完成，主循环开始 ...", flush=True)

        # 建立函数名→源文件反查表：扫描 --src 下所有 .c 文件，用正则找函数定义
        # 当 DWARF 无法给出有效 .c 时，用此表回退
        _func_to_src: Dict[str, str] = {}
        if not args.skip_alignment:
            import os
            _fn_def_re = re.compile(
                r'(?:^|\n)\s*(?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:unsigned\s+)?'
                r'(?:struct\s+\w+\s*\*?\s*|enum\s+\w+\s+|union\s+\w+\s+|'
                r'void|int|long|char|short|float|double|size_t|ssize_t|bool|'
                r'CURLcode|CURLMcode|CURLSHcode|CURLMsg|CURL|CURLM|CURLSH|'
                r'\w+_t|\w+)'
                r'\s*\*{0,3}\s*(\w+)\s*\('
            )
            for root_dir, dirs, files in os.walk(str(src_root)):
                dirs[:] = [d for d in dirs if d not in ('.git', '.bc_o0', 'tests', 'docs')]
                for fname in files:
                    if not fname.endswith('.c'):
                        continue
                    fpath = Path(root_dir) / fname
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        ck = _src_cache_key(fpath)
                        for m in _fn_def_re.finditer(content):
                            fn = m.group(1)
                            if fn not in _func_to_src:
                                _func_to_src[fn] = ck
                        if ck not in _src_cache:
                            ll = content.splitlines()
                            lt = _LineLookup(ll)
                            _src_cache[ck] = (ll, lt)
                        if _HAS_TREE_SITTER and ck not in _stmt_parse_cache:
                            stmts, lts = _parse_snippet_and_offset(content, 0)
                            _stmt_parse_cache[ck] = (stmts, lts)
                    except Exception:
                        pass
            print(f"  函数名→源文件反查表: {len(_func_to_src)} 个函数定义", flush=True)

        _src_root_resolved = Path(_src_cache_key(src_root))

        def _source_to_ll_stem(source_rel):
            if not source_rel:
                return None
            s = source_rel.lstrip('./')
            s = s.replace(os.sep, '_').replace('/', '_')
            if s.endswith('.c'):
                s = s[:-2]
            return s if s in ir_dir_stems else None

        def _parse_ir_cached(stem):
            if stem not in _ir_dir_cache:
                ll_p = ir_dir / f"{stem}.ll"
                _ir_dir_cache[stem] = parse_llvm_ir_alignment(ll_p) if ll_p.exists() else {}
            return _ir_dir_cache[stem]

        def _lookup_ir(fname, source_rel=None):
            """查找函数的 IR BB map：单文件模式或逐文件目录模式"""
            if ir_functions:
                return ir_functions.get(fname)
            if not ir_dir:
                return None
            if source_rel:
                stem = _source_to_ll_stem(source_rel)
                if stem:
                    r = _parse_ir_cached(stem).get(fname)
                    if r:
                        return r
            src_abs = _func_to_src.get(fname)
            if src_abs:
                try:
                    rel = str(Path(src_abs).relative_to(_src_root_resolved))
                except ValueError:
                    return None
                stem = _source_to_ll_stem(rel)
                if stem:
                    return _parse_ir_cached(stem).get(fname)
            return None

        print(f"  共 {total_funcs} 个函数，开始处理 ...", flush=True)

        for func_name, addr, size in func_list:
            total += 1
            _sig_stats: Dict[str, int] = {}
            if total % 200 == 0 or total == 1:
                print(f"  进度 {total}/{total_funcs}, 写出 {written} 当前函数={func_name[:40]}", flush=True)

            start = addr - base_addr
            if start < 0 or start + size > len(text_bytes):
                continue
            code_bytes = text_bytes[start : start + size]
            cfg, block_addr_ranges = build_cfg_for_function(
                code_bytes, addr, func_name, num_opcodes=args.num_opcodes
            )
            if cfg.num_blocks == 0:
                continue

            ir_bb_map = _lookup_ir(func_name)
            mach_to_ir = {}
            if ir_bb_map:
                mach_to_ir = map_machine_bb_to_ir(
                    block_addr_ranges, dwarf_entries, get_lines, ir_bb_map
                )

            block_lines, block_confidence = cross_validate(
                block_addr_ranges, dwarf_entries, get_lines, ir_bb_map, mach_to_ir
            )

            all_lines = [ln for ls in block_lines for ln in ls]
            primary_path = None
            prim_fi = None
            prim_file_path = None
            prim_file_rel = None
            ordered_sources: List[Tuple[str, str]] = []
            seen_sources: Set[Tuple[str, str]] = set()
            lines_list: List[str] = []
            line_to_text: Dict[int, str] = {}
            src_content = ""
            this_func_had_src = False
            if block_lines and any(block_lines):
                # 收集本函数所有 BB 出现过的源文件，优先 .c 文件（避免头文件无语句导致回退）
                for sa, ea in block_addr_ranges:
                    for e in get_lines(sa, ea + 8, dwarf_entries):
                        fp = e.get("file_path")
                        fr = e.get("file_path_rel") or ""
                        if fp:
                            seen_sources.add((fp, fr))
                # 按路径后缀排序：.c 优先，再 .h，再其它
                def _path_prio(item: Tuple[str, str]) -> tuple:
                    path = item[0] or ""
                    p = Path(path)
                    if p.suffix == ".c":
                        return (0, path)
                    if p.suffix in (".h", ".hpp"):
                        return (1, path)
                    return (2, path)
                ordered_sources = sorted(seen_sources, key=_path_prio)
                _need_min = min(all_lines) if all_lines else 0
                _need_max = max(all_lines) if all_lines else 0
                for raw_path, rel_path in ordered_sources:
                    primary_path = raw_path
                    p = Path(primary_path)
                    if p.suffix not in (".c", ".h", ".hpp", ".cpp"):  # DWARF 有时把目录当 file，跳过
                        continue
                    cands = [p, comp_dir / p.name, comp_dir / p]
                    if rel_path and Path(rel_path).suffix in (".c", ".h", ".hpp", ".cpp"):
                        cands.append(comp_dir / rel_path)
                        cands.append(comp_dir / Path(rel_path).name)  # 仅文件名
                    # 注：rglob 已移除，561 文件 × 递归遍历会导致极慢（每函数数秒）
                    for c in cands:
                        if not c.exists() or not c.is_file():
                            continue
                        ck = _src_cache_key(c)
                        if ck in _src_cache:
                            ll, lt = _src_cache[ck]
                            n_lines = len(ll)
                        else:
                            try:
                                raw = c.read_text(encoding="utf-8", errors="replace")
                                ll = raw.splitlines()
                                n_lines = len(ll)
                                lt = _LineLookup(ll)
                                _src_cache[ck] = (ll, lt)
                            except Exception:
                                continue
                        # 只接受“包含本函数行号范围”的文件，避免错用别文件导致无重叠语句
                        # 放宽 _need_max 容差（行尾/编码差异），_need_min >= 1 避免空文件
                        if _need_max <= n_lines + 20 and _need_min >= 1:
                            cache_key = ck
                            lines_list = ll
                            line_to_text = lt
                            src_content = "\n".join(ll)
                            prim_file_path = raw_path
                            prim_file_rel = rel_path or None
                            this_func_had_src = True
                            break
                    if this_func_had_src:
                        break
                # 仅用主文件的行号做 tree-sitter 范围，避免头文件行号与主文件混合
                if this_func_had_src and (prim_file_path or prim_file_rel):
                    all_lines_prim = []
                    for sa, ea in block_addr_ranges:
                        for e in get_lines(sa, ea + 8, dwarf_entries):
                            if (prim_file_path and e.get("file_path") == prim_file_path) or (
                                prim_file_rel and e.get("file_path_rel") == prim_file_rel
                            ):
                                all_lines_prim.append(e["line"])
                    if all_lines_prim:
                        all_lines = all_lines_prim

            # 当 DWARF 找不到有效 .c 时，用函数名反查表回退：
            # 找到源文件 + 定位函数体行范围 → 直接构建行级对齐（不经过 generate_alignment_ground_truth，
            # 因为 DWARF 行号来自系统头，和实际 .c 对不上）
            _used_func_fallback = False
            if not this_func_had_src and not args.skip_alignment and func_name in _func_to_src:
                ck = _func_to_src[func_name]
                if ck in _src_cache:
                    fb_lines_list, fb_line_to_text = _src_cache[ck]
                    fb_src_content = "\n".join(fb_lines_list)
                    _func_def_pattern = re.compile(
                        r'(?:^|\n)[^\n]*\b' + re.escape(func_name) + r'\s*\(', re.MULTILINE
                    )
                    _m = _func_def_pattern.search(fb_src_content)
                    if _m:
                        def_line = fb_src_content[:_m.end()].count('\n') + 1
                        brace_depth = 0
                        func_end_line = def_line
                        started = False
                        for li, ltxt in enumerate(fb_lines_list[def_line - 1:], def_line):
                            for ch in ltxt:
                                if ch == '{':
                                    brace_depth += 1
                                    started = True
                                elif ch == '}':
                                    brace_depth -= 1
                            if started and brace_depth <= 0:
                                func_end_line = li
                                break
                        if func_end_line <= def_line:
                            func_end_line = min(def_line + 100, len(fb_lines_list))
                        fb_all_lines = list(range(def_line, func_end_line + 1))

                        _fb_ok = False
                        if ck in _stmt_parse_cache:
                            stmts_all, lts_all = _stmt_parse_cache[ck]
                            fb_stmts_raw = [s for s in stmts_all
                                            if s["end_line"] >= def_line and s["start_line"] <= func_end_line]
                            if fb_stmts_raw:
                                o2n: Dict[int, int] = {}
                                fb_stmts: List[Dict] = []
                                for s in fb_stmts_raw:
                                    nid = len(fb_stmts)
                                    o2n[s["id"]] = nid
                                    fb_stmts.append({**s, "id": nid})
                                fb_lts: Dict[int, List[Dict]] = {}
                                for ln in range(def_line, func_end_line + 1):
                                    for s in lts_all.get(ln, []):
                                        if s["id"] in o2n:
                                            fb_lts.setdefault(ln, []).append(fb_stmts[o2n[s["id"]]])
                                ir_map_func = _lookup_ir(func_name, prim_file_rel)
                                if ir_map_func and fb_lts:
                                    ir_sorted = sorted(
                                        [(l, info) for l, info in ir_map_func.items() if info.get("lines")],
                                        key=lambda x: min(x[1]["lines"])
                                    )
                                    n_b = len(block_addr_ranges)
                                    n_s = len(fb_stmts)
                                    M_gt_fb = [[0.0] * n_s for _ in range(n_b)]
                                    W_fb = [[0.0] * n_s for _ in range(n_b)]
                                    fb_bl = [[] for _ in range(n_b)]
                                    for b_id in range(n_b):
                                        ir_idx = min(b_id, len(ir_sorted) - 1) if ir_sorted else -1
                                        if ir_idx >= 0:
                                            ir_ls = ir_sorted[ir_idx][1]["lines"]
                                            in_r = {l for l in ir_ls if def_line <= l <= func_end_line}
                                            fb_bl[b_id] = sorted(in_r)
                                            if in_r:
                                                sid, conf = map_lines_to_statement(in_r, fb_lts, fb_stmts)
                                                if sid is not None:
                                                    M_gt_fb[b_id][sid] = 1.0
                                                    W_fb[b_id][sid] = conf * 0.8
                                    for i in range(n_b):
                                        rs = sum(M_gt_fb[i])
                                        if rs > 0:
                                            for j in range(n_s):
                                                M_gt_fb[i][j] /= rs
                                    alignment_matrix = M_gt_fb
                                    alignment_confidence = W_fb
                                    block_lines = fb_bl
                                    _fb_ok = True
                        if not _fb_ok:
                            # 勿仅用「函数体内源码行均分」到 n_bb：当 O0/O3 的 Capstone BB 数相同时，
                            # 分块完全相同 → alignment_confidence 与优化档无关，易出现四档 W 逐位一致。
                            # 优先用本 ELF 上 cross_validate 的 DWARF(+IR) 行与置信度。
                            used_dwarf_lines = False
                            dwarf_bl: List[List[int]] = []
                            dwarf_bc: List[List[float]] = []
                            for ls, bc in zip(block_lines, block_confidence):
                                if ls:
                                    dwarf_bl.append(list(ls))
                                    dwarf_bc.append(list(bc))
                            if dwarf_bl and len(dwarf_bc) == len(dwarf_bl):
                                flat_u = sorted({ln for ls in dwarf_bl for ln in ls})
                                if flat_u:
                                    idx_map = {ln: j for j, ln in enumerate(flat_u)}
                                    alignment_matrix, alignment_confidence = build_alignment_matrix(
                                        dwarf_bl, idx_map, dwarf_bc
                                    )
                                    block_lines = dwarf_bl
                                    used_dwarf_lines = True
                            if not used_dwarf_lines:
                                fb_line_to_stmt_idx = {ln: i for i, ln in enumerate(fb_all_lines)}
                                fb_block_lines: List[List[int]] = []
                                n_fb_blocks = len(block_addr_ranges)
                                per_block = max(1, len(fb_all_lines) // max(n_fb_blocks, 1))
                                for b_id in range(n_fb_blocks):
                                    si = b_id * per_block
                                    ei = si + per_block if b_id < n_fb_blocks - 1 else len(fb_all_lines)
                                    fb_block_lines.append(fb_all_lines[si:ei])
                                alignment_matrix, alignment_confidence = build_alignment_matrix(
                                    fb_block_lines, fb_line_to_stmt_idx
                                )
                                block_lines = fb_block_lines
                        lines_list = fb_lines_list
                        line_to_text = fb_line_to_text
                        src_content = fb_src_content
                        cache_key = ck
                        primary_path = ck
                        prim_file_path = ck
                        all_lines = fb_all_lines
                        this_func_had_src = True
                        _used_func_fallback = True
                        n_tree_sitter += 1

            if not this_func_had_src and not _used_func_fallback:
                n_skipped_no_src += 1
                continue

            import torch
            with torch.no_grad():
                struct_feats = extract_structural_features(cfg)
                pattern_labels = assign_pattern_labels(cfg)

            source_statements_used: List[Dict] = []
            func_min_ln = min(all_lines) if all_lines else 0
            func_max_ln = max(all_lines) if all_lines else 0
            stmt_cache_key: Optional[str] = None
            if not _used_func_fallback:
                alignment_matrix: List[List[float]] = []
                alignment_confidence: List[List[float]] = []
                _sig_stats: Dict[str, int] = {}
            line_nums: List[int] = []

            if not _used_func_fallback and not args.skip_alignment and _HAS_TREE_SITTER and src_content and lines_list and all_lines:
                stmt_cache_key = cache_key
                if stmt_cache_key and stmt_cache_key in _stmt_parse_cache:
                    statements_full, line_to_stmts_full = _stmt_parse_cache[stmt_cache_key]
                else:
                    if not _stmt_parse_cache:
                        print("  解析源文件 (大文件约 30-90s，请稍候)...", flush=True)
                    statements_full, line_to_stmts_full = _parse_snippet_and_offset(src_content, 0)
                    if stmt_cache_key:
                        _stmt_parse_cache[stmt_cache_key] = (statements_full, line_to_stmts_full)
                    if (args.verbose or not _ts_debug_printed):
                        n_lines = len(src_content.splitlines())
                        types = list(set(s["type"] for s in statements_full))[:10]
                        print(f"  源文件解析完成: {len(statements_full)} 语句 (约 {n_lines} 行)", flush=True)
                        _ts_debug_printed = True
                filtered_stmts = []
                orig_id_to_new: Dict[int, int] = {}
                # 先按原始范围过滤
                for s in statements_full:
                    if s["end_line"] >= func_min_ln and s["start_line"] <= func_max_ln:
                        new_id = len(filtered_stmts)
                        orig_id_to_new[s["id"]] = new_id
                        filtered_stmts.append({**s, "id": new_id})
                # 无重叠时放宽范围 ±15 行重试（DWARF 行号有时有偏差）
                if not filtered_stmts and statements_full and func_min_ln > 0 and func_max_ln > 0:
                    expand = 15
                    fmin, fmax = max(1, func_min_ln - expand), func_max_ln + expand
                    for s in statements_full:
                        if s["end_line"] >= fmin and s["start_line"] <= fmax:
                            new_id = len(filtered_stmts)
                            orig_id_to_new[s["id"]] = new_id
                            filtered_stmts.append({**s, "id": new_id})
                if args.verbose and not filtered_stmts and statements_full and n_fallback < 3:
                    stmt_ln_range = (min(s["start_line"] for s in statements_full), max(s["end_line"] for s in statements_full))
                    n_overlap = sum(1 for s in statements_full if s["end_line"] >= func_min_ln and s["start_line"] <= func_max_ln)
                    print(f"  [DEBUG] {func_name}: 无重叠, dwarf[{func_min_ln},{func_max_ln}] stmt{stmt_ln_range} 重叠数={n_overlap} prim={primary_path}", flush=True)
                if filtered_stmts:
                    line_to_stmts: Dict[int, List[Dict]] = {}
                    stmt_by_id = {s["id"]: s for s in filtered_stmts}
                    stmt_ln_min = min(s["start_line"] for s in filtered_stmts)
                    stmt_ln_max = max(s["end_line"] for s in filtered_stmts)
                    for line in line_to_stmts_full:
                        if not (stmt_ln_min <= line <= stmt_ln_max):
                            continue
                        for stmt in line_to_stmts_full[line]:
                            if stmt["id"] in orig_id_to_new:
                                new_id = orig_id_to_new[stmt["id"]]
                                ref = stmt_by_id.get(new_id)
                                if ref is not None:
                                    line_to_stmts.setdefault(line, []).append(ref)
                    machine_bbs = [
                        {"addr": sa, "size": max(ea - sa + 1, 1)}
                        for sa, ea in block_addr_ranges
                    ]
                    ir_map = _lookup_ir(func_name, prim_file_rel)
                    alignment_matrix, alignment_confidence, _sig_stats = generate_alignment_ground_truth(
                        machine_bbs, filtered_stmts, dwarf_entries, ir_map, line_to_stmts, get_lines,
                        prim_file_path=prim_file_path,
                        prim_file_rel=prim_file_rel,
                    )
                    source_statements_used = filtered_stmts
                    line_nums = sorted(set(ln for s in filtered_stmts for ln in range(s["start_line"], s["end_line"] + 1)))
                    n_tree_sitter += 1

            if not alignment_matrix:
                line_nums = sorted(set(all_lines))
                line_to_stmt_idx = {ln: i for i, ln in enumerate(line_nums)}
                alignment_matrix, alignment_confidence = build_alignment_matrix(block_lines, line_to_stmt_idx, block_confidence)
                if ir_bb_map and block_confidence:
                    n_both = n_dwarf_only = n_ir_only = n_none = 0
                    n_agree = n_near = n_conflict = 0
                    for b_id, (sa, ea) in enumerate(block_addr_ranges):
                        d_info = get_lines(sa, ea + 8, dwarf_entries)
                        d_lines = {x["line"] for x in d_info}
                        i_lines: Set[int] = set()
                        if ir_bb_map and b_id in mach_to_ir:
                            ir_label, _ = mach_to_ir[b_id]
                            ii = ir_bb_map.get(ir_label)
                            if ii:
                                i_lines = ii.get("lines") or set()
                        if d_lines and i_lines:
                            n_both += 1
                            if d_lines == i_lines:
                                n_agree += 1
                            elif d_lines & i_lines:
                                n_near += 1
                            else:
                                n_conflict += 1
                        elif d_lines:
                            n_dwarf_only += 1
                        elif i_lines:
                            n_ir_only += 1
                        else:
                            n_none += 1
                    _sig_stats = {
                        "n_both": n_both, "n_dwarf_only": n_dwarf_only,
                        "n_ir_only": n_ir_only, "n_none": n_none,
                        "n_agree": n_agree, "n_near": n_near, "n_conflict": n_conflict,
                    }
                n_fallback += 1
                if not this_func_had_src:
                    n_fallback_no_src += 1
                    if len(diag_no_src) < 5:
                        cands_exist: List[Tuple[str, bool]] = []
                        for path_str, rel_path in list(ordered_sources)[:3]:
                            c = Path(path_str)
                            ex = (c.exists() and c.is_file()) or ((comp_dir / c.name).exists() and (comp_dir / c.name).is_file()) or ((comp_dir / c).exists() and (comp_dir / c).is_file())
                            if rel_path:
                                rp = comp_dir / rel_path
                                ex = ex or (rp.exists() and rp.is_file())
                            cands_exist.append((path_str[:60], ex))
                        diag_no_src.append((func_name[:40], not seen_sources, len(seen_sources), cands_exist))
                else:
                    n_fallback_no_stmts += 1
                    if len(diag_no_stmts) < 5 and primary_path is not None:
                        stmt_range: Optional[Tuple[int, int]] = None
                        if stmt_cache_key and stmt_cache_key in _stmt_parse_cache:
                            sts, _ = _stmt_parse_cache[stmt_cache_key]
                            if sts:
                                stmt_range = (min(s["start_line"] for s in sts), max(s["end_line"] for s in sts))
                        diag_no_stmts.append((func_name[:40], primary_path[:60] if primary_path else None, func_min_ln, func_max_ln, stmt_range))

            bin_opcodes, bin_ot, bin_ov, bin_bl = [], [], [], []
            for bb in cfg.blocks:
                bin_bl.append(len(bb.instructions))
                for inst in bb.instructions:
                    bin_opcodes.append(inst.opcode)
                    ot = (inst.operand_types + [0, 0, 0])[:3]
                    vb = (inst.value_buckets + [0, 0, 0])[:3]
                    bin_ot.extend(ot)
                    bin_ov.extend(vb)
            src_edges = [list(e[0] for e in cfg.edges), list(e[1] for e in cfg.edges)]
            etype = [e[2] for e in cfg.edges]

            n_cols = len(alignment_matrix[0]) if alignment_matrix else 0
            stmt_lens = []
            tok_ids = []
            line_nums_flat: List[int] = []
            # 训练侧可据此过滤：stmt=tree-sitter 语句列；line_fallback=按行编码且无 src_edge
            src_encoding_tier = "line_fallback"
            if source_statements_used:
                src_encoding_tier = "stmt"
                # 列压缩：只保留有非零对齐的语句列，大幅减小 SQLite 等大文件项目的体积
                used_cols: List[int] = []
                if alignment_matrix and n_cols > 0:
                    for j in range(n_cols):
                        if any(alignment_matrix[i][j] > 0.001 for i in range(len(alignment_matrix))):
                            used_cols.append(j)
                else:
                    used_cols = list(range(len(source_statements_used)))

                if len(used_cols) < len(source_statements_used):
                    source_statements_used = [source_statements_used[j] for j in used_cols]
                    alignment_matrix = [[alignment_matrix[i][j] for j in used_cols] for i in range(len(alignment_matrix))]
                    alignment_confidence = [[alignment_confidence[i][j] for j in used_cols] for i in range(len(alignment_confidence))]

                sync_block_src_lines_with_alignment_matrix(
                    block_lines, block_confidence, alignment_matrix
                )

                for stmt in source_statements_used:
                    t = "\n".join(
                        line_to_text.get(ln, "")
                        for ln in range(stmt["start_line"], stmt["end_line"] + 1)
                    )
                    ids = [stable_embedding_id(w, args.src_vocab_size) for w in t.split()] or [0]
                    stmt_lens.append(len(ids))
                    tok_ids.extend(ids)
                line_nums_flat = sorted(set(
                    ln for s in source_statements_used
                    for ln in range(s["start_line"], s["end_line"] + 1)
                ))
                ast_type_map = {
                    "if_statement": 2, "while_statement": 3, "for_statement": 4,
                    "do_statement": 5, "switch_statement": 6, "case_statement": 7,
                    "return_statement": 8, "break_statement": 9, "continue_statement": 10,
                    "goto_statement": 11, "compound_statement": 12,
                }
                ast_types = [ast_type_map.get(s["type"], 15) for s in source_statements_used]

                # 利用 BB↔语句对齐 + CFG 结构，对 pattern_labels 做细化：
                #  - IF_HEADER: 有 COND 出边且对齐 if_statement 的块
                #  - IF_THEN_BODY: IF_HEADER 的后继且对齐同一 if_statement
                #  - IF_ELSE_BODY: IF_HEADER 的后继、对齐非 if 语句、仅单前驱（非合并点）
                #  - LOOP_HEADER/SWITCH_HEADER/SWITCH_CASE_BODY/FUNC_PROLOGUE_EPILOGUE 同前
                if alignment_matrix and pattern_labels is not None and cfg.edges:
                    n_bb_rows = len(alignment_matrix)
                    n_stmt_cols = len(alignment_matrix[0]) if n_bb_rows > 0 else 0
                    if n_bb_rows == len(pattern_labels) and n_stmt_cols == len(ast_types):
                        bb2stmt = {}
                        for bi in range(n_bb_rows):
                            row = alignment_matrix[bi]
                            if row and max(row) >= 0.1:
                                bb2stmt[bi] = row.index(max(row))
                        pred_count = [0] * n_bb_rows
                        for s, d, _ in cfg.edges:
                            pred_count[d] += 1
                        cond_src = set()
                        for s, d, et in cfg.edges:
                            if et == int(EdgeType.COND):
                                cond_src.add(s)
                        for bi in range(n_bb_rows):
                            stmt_j = bb2stmt.get(bi, -1)
                            if stmt_j < 0:
                                continue
                            t = ast_types[stmt_j]
                            cur = pattern_labels[bi]
                            if cur != int(PatternType.SEQUENTIAL):
                                continue
                            if t == 2:  # if_statement
                                if bi in cond_src:
                                    pattern_labels[bi] = int(PatternType.IF_HEADER)
                                else:
                                    preds = [s for s, d, _ in cfg.edges if d == bi]
                                    if len(preds) == 1 and preds[0] in cond_src and bb2stmt.get(preds[0]) == stmt_j:
                                        pattern_labels[bi] = int(PatternType.IF_THEN_BODY)
                            elif t in (3, 4, 5):  # while / for / do
                                pattern_labels[bi] = int(PatternType.LOOP_HEADER)
                            elif t == 6:  # switch
                                pattern_labels[bi] = int(PatternType.SWITCH_HEADER)
                            elif t == 7:  # case
                                pattern_labels[bi] = int(PatternType.SWITCH_CASE_BODY)
                            elif t in (8, 9, 10, 11):  # return / break / continue / goto
                                if not any(e[0] == bi for e in cfg.edges):
                                    pattern_labels[bi] = int(PatternType.FUNC_PROLOGUE_EPILOGUE)
                        for bi in range(n_bb_rows):
                            if pattern_labels[bi] != int(PatternType.IF_HEADER):
                                continue
                            header_stmt = bb2stmt.get(bi, -1)
                            if header_stmt < 0 or ast_types[header_stmt] != 2:
                                continue
                            succs = [d for s, d, _ in cfg.edges if s == bi]
                            for sj in succs:
                                if pattern_labels[sj] != int(PatternType.SEQUENTIAL):
                                    continue
                                stmt_sj = bb2stmt.get(sj, -1)
                                if stmt_sj < 0:
                                    continue
                                if stmt_sj == header_stmt:
                                    pattern_labels[sj] = int(PatternType.IF_THEN_BODY)
                                elif pred_count[sj] == 1:
                                    # 早退分支：对齐 return/break/continue/goto 且后继唯一且后继无出边 → THEN
                                    if ast_types[stmt_sj] in (8, 9, 10, 11):
                                        sj_succs = [d for s, d, _ in cfg.edges if s == sj]
                                        if len(sj_succs) == 1:
                                            succ_out = [e for e in cfg.edges if e[0] == sj_succs[0]]
                                            if len(succ_out) == 0:
                                                pattern_labels[sj] = int(PatternType.IF_THEN_BODY)
                                                continue
                                    pattern_labels[sj] = int(PatternType.IF_ELSE_BODY)

                src_edge_to_et: Dict[Tuple[int, int], int] = {}
                if alignment_matrix and cfg.edges:
                    n_bb_rows = len(alignment_matrix)
                    n_stmt_cols = len(alignment_matrix[0]) if n_bb_rows > 0 else 0
                    if n_stmt_cols > 0:
                        bb2stmt = {}
                        for bi in range(n_bb_rows):
                            row = alignment_matrix[bi]
                            mx = max(row)
                            if mx > 0.001:
                                bb2stmt[bi] = row.index(mx)
                        for sb, db, et in cfg.edges:
                            ss = bb2stmt.get(sb, -1)
                            ds = bb2stmt.get(db, -1)
                            if ss >= 0 and ds >= 0 and ss != ds:
                                key = (ss, ds)
                                if key not in src_edge_to_et:
                                    src_edge_to_et[key] = int(et)
                src_edge_sorted = sorted(src_edge_to_et.keys())
                src_edge_idx = (
                    [[e[0] for e in src_edge_sorted], [e[1] for e in src_edge_sorted]]
                    if src_edge_sorted
                    else [[], []]
                )
                src_edge_types = [src_edge_to_et[k] for k in src_edge_sorted]
            else:
                sync_block_src_lines_with_alignment_matrix(
                    block_lines, block_confidence, alignment_matrix
                )
                line_nums_flat = sorted(set(ln for ls in block_lines for ln in ls))
                for ln in line_nums_flat:
                    t = line_to_text.get(ln, "")
                    ids = [stable_embedding_id(w, args.src_vocab_size) for w in t.split()] or [0]
                    stmt_lens.append(len(ids))
                    tok_ids.extend(ids)
                ast_types = [15] * len(line_nums_flat)
                src_edge_idx = [[], []]
                src_edge_types = []

            if src_encoding_tier == "stmt" and not stmt_lens:
                src_encoding_tier = "line_fallback"

            n_bb_bl = len(block_lines)
            bb_alignment_coverage = 0.0
            if n_bb_bl > 0 and alignment_matrix and len(alignment_matrix) >= n_bb_bl:
                bb_alignment_coverage = (
                    sum(
                        1
                        for i in range(n_bb_bl)
                        if alignment_matrix[i] and sum(alignment_matrix[i]) > 1e-12
                    )
                    / n_bb_bl
                )

            if not args.keep_line_fallback and src_encoding_tier == "line_fallback":
                n_skipped_line_fallback += 1
                continue

            sample = {
                "binary": str(elf_path),
                "func_name": func_name,
                "opt_level": args.opt,
                "bin_opcodes": bin_opcodes,
                "bin_operand_types": bin_ot,
                "bin_operand_values": bin_ov,
                "bin_instr_text": [
                    [inst.raw_text for inst in block.instructions]
                    for block in cfg.blocks
                ],
                "bin_block_lengths": bin_bl,
                "bin_edge_index": src_edges,
                "bin_edge_type": etype,
                "bin_struct_features": struct_feats.tolist(),
                "src_token_ids": tok_ids,
                "src_stmt_lengths": stmt_lens,
                "src_edge_index": src_edge_idx,
                "src_edge_type": src_edge_types,
                "src_ast_types": ast_types,
                "block_src_lines": block_lines,
                "block_src_confidence": block_confidence,
                "pattern_labels": pattern_labels,
                "source_file": str(primary_path) if primary_path else None,
                "source_lines": line_nums_flat,
                "src_encoding_tier": src_encoding_tier,
                "bb_alignment_coverage": round(bb_alignment_coverage, 6),
            }
            if not args.skip_alignment:
                sample["alignment_matrix"] = alignment_matrix
                sample["alignment_confidence"] = alignment_confidence
                if _sig_stats:
                    sample["signal_stats"] = _sig_stats
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"写出: {out_path}")
    print(f"样本数: {written}")
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"文件大小: {size_mb:.1f} MB")
    if ir_dir and _ir_dir_cache:
        n_ir_files_used = sum(1 for v in _ir_dir_cache.values() if v)
        n_ir_funcs = sum(len(v) for v in _ir_dir_cache.values())
        print(f"逐文件 IR: 使用 {n_ir_files_used}/{len(ir_dir_stems)} 个 .ll, 解析 {n_ir_funcs} 个 IR 函数")
    if n_skipped_no_src > 0:
        print(f"跳过无源码函数: {n_skipped_no_src} (静态链接的库函数)")
    if n_skipped_line_fallback > 0:
        print(f"跳过行级回退样本(未写出): {n_skipped_line_fallback} (--keep-line-fallback 可保留)")
    if total > 0:
        if args.skip_alignment:
            print("对齐方式: 快速模式(全行级)")
        else:
            print(f"对齐方式: tree-sitter(语句级) {n_tree_sitter}, 行级回退 {n_fallback}")
            if n_fallback > 0:
                print(f"  回退原因: 未解析到源码 {n_fallback_no_src}, 无重叠语句 {n_fallback_no_stmts}")
                if diag_no_src:
                    print("  诊断 [未解析到源码] 样例 (函数, seen_fi空?, seen_fi前5, (fi, path, 存在?)):")
                    for t in diag_no_src:
                        print(f"    {t}")
                if diag_no_stmts:
                    print("  诊断 [无重叠语句] 样例 (函数, 主文件, dwarf行范围, 文件stmt行范围):")
                    for t in diag_no_stmts:
                        print(f"    {t}")


if __name__ == "__main__":
    main()
