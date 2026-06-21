#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 binskel 单条样本的基本块对齐做 **双通道监测**（与「金标准正确」无关，仅为自动化启发式）：

**结构监测**（矩阵 + CFG 自洽，不读源码语义）：
  - M_gt 与 block_src_lines：有 DWARF 行的 BB 是否都有非零对齐行
  - BB 覆盖率、空行 BB 比例、相邻 BB 主峰语句列顺序一致性、COND 边一致性（来自 validate_sample / analyze_sample）

**语义监测**（读源码 + tree-sitter，启发式）：
  - check_alignment_semantics：主峰列推断的语句行是否与该 BB 的 DWARF 行有交；高 W 单元是否一致
  - 加 `--sem-asm-grounded` +（可选）`--bin-root`：在 ELF 上 **Capstone 重切 BB**，用 **.debug_line 按 PC 重放** 行号再核对（强于仅用 json 内行号，仍非形式化证明）
  - 输出按 BB 计的 viol_peak / viol_highw

「对不对」：仅表示 **是否通过上述检查**；通过 ≠ 形式化正确。

用法:
  python3 scripts/monitor_bb_alignment_dual.py \\
    /path/binskel_zlib_o0.jsonl --src-root /path/zlib_build --limit 50

  python3 scripts/monitor_bb_alignment_dual.py --scan \\
    --out-dir ~/structlift_datasets --all-default-paths \\
    --opt-levels 0,1,2,3,os --per-file 200

  # 45 个数据集各随机抽 K 条（蓄水池），做结构+语义判定，CSV 输出：
  python3 scripts/monitor_bb_alignment_dual.py --scan --all-default-paths \\
    --out-dir ~/structlift_datasets --opt-levels 0,1,2,3,os \\
    --random-per-file 5 --seed 42 --csv > bb_dual_random.csv

  # 同上，但 **每个基本块一行**（--by-bb），语义 violation 为完整 BB 列表：
  python3 scripts/monitor_bb_alignment_dual.py --scan --all-default-paths \\
    --by-bb --out-dir ~/structlift_datasets --opt-levels 0,1,2,3,os \\
    --random-per-file 5 --seed 42 --csv > bb_dual_by_bb.csv

  # 语义改为 **ELF+PC 重放 DWARF**（强于仅 json 行号），binary 路径不对时加 --bin-root：
  python3 scripts/monitor_bb_alignment_dual.py --scan --all-default-paths \\
    --sem-asm-grounded --bin-root ~/structlift_datasets \\
    --by-bb --out-dir ~/structlift_datasets --opt-levels 0,1,2,3,os \\
    --random-per-file 5 --seed 42 --csv > bb_dual_by_bb_asm.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deep_validate import analyze_sample  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402

try:
    from check_alignment_semantics import check_sample as semantic_check_sample  # noqa: E402
except ImportError:
    semantic_check_sample = None  # type: ignore

try:
    from binskel_alignment_batch_stats import (  # noqa: E402
        DEFAULT_PROJECTS,
        dedupe_paths,
        locate_jsonl,
        default_multi_root_search_list,
    )
except ImportError:
    DEFAULT_PROJECTS = []
    dedupe_paths = None  # type: ignore
    locate_jsonl = None  # type: ignore
    default_multi_root_search_list = None  # type: ignore


def binskel_jsonl_name(proj: str, opt_token: str) -> str:
    t = opt_token.strip().lower()
    if t == "os":
        return f"binskel_{proj}_os.jsonl"
    return f"binskel_{proj}_o{t}.jsonl"


def reservoir_sample_dicts(path: Path, k: int, rng: random.Random) -> List[Dict[str, Any]]:
    """单遍扫描 jsonl，随机保留 k 条有效 JSON（蓄水池）；k=0 表示不抽样。"""
    if k <= 0:
        return []
    reservoir: List[Dict[str, Any]] = []
    i = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i < k:
                reservoir.append(d)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = d
            i += 1
    return reservoir


def parse_opt_levels(s: str) -> List[str]:
    out: List[str] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p.lower() if p.lower() == "os" else p.lstrip("oO") or "0")
    return out if out else ["0"]


def structural_grade(
    d: Dict[str, Any], v: Dict[str, Any], r_val: Dict[str, float]
) -> Tuple[str, List[str]]:
    notes: List[str] = []
    g = "OK"
    if not v.get("matrix_lines_consistent", True):
        g = "FAIL"
        notes.append("M 与 block_src_lines 不自洽(有行无对齐)")
    if v.get("empty_ratio", 0) > 0.2:
        g = "WARN" if g != "FAIL" else g
        notes.append(f"空对齐BB比例={v['empty_ratio']:.1%}")
    if v.get("order_consistency", 1) < 0.7:
        g = "WARN" if g != "FAIL" else g
        notes.append(f"顺序一致性={v['order_consistency']:.2f}")
    if r_val.get("coverage", 0) < 0.8:
        g = "WARN" if g != "FAIL" else g
        notes.append(f"validate覆盖率={r_val['coverage']:.2f}")
    return g, notes


def semantic_grade(sem: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not sem:
        return "SKIP", ["未配置语义检查"]
    if sem.get("skip"):
        return "SKIP", [sem.get("reason", "skip")]
    notes: List[str] = []
    g = "OK"
    npk = int(sem.get("n_viol_peak") or 0)
    nhw = int(sem.get("n_viol_highw") or 0)
    nchk = int(sem.get("n_bb_checked") or 0)
    if npk > 0:
        g = "WARN"
        notes.append(f"主峰启发式可疑 BB 数={npk}/{max(nchk,1)}")
    if nhw > 0:
        g = "WARN"
        notes.append(f"高W启发式可疑单元={nhw}")
    if sem.get("weak_map_cols", 0) > max(3, (sem.get("n_col") or 0) // 15):
        g = "WARN" if g == "OK" else g
        notes.append(f"弱列映射={sem.get('weak_map_cols')}")
    if sem.get("asm_grounded"):
        if not sem.get("asm_cfg_ok", True):
            g = "WARN" if g == "OK" else g
            notes.append("ELF重切与bin_block_lengths等不一致")
        ndj = int(sem.get("n_viol_dwarf_json") or 0)
        if ndj > 0:
            g = "WARN" if g == "OK" else g
            notes.append(f"PC重放行号 vs json 行号 可疑BB={ndj}")
        npc = int(sem.get("n_viol_no_pc_line") or 0)
        if npc > 0:
            g = "WARN" if g == "OK" else g
            notes.append(f"有json行号但PC无行号 BB 数={npc}")
    return g, notes


def process_one(
    d: Dict[str, Any],
    src_root: Optional[Path],
    high_w: float,
    *,
    asm_grounded: bool = False,
    bin_root: Optional[Path] = None,
) -> Dict[str, Any]:
    v = analyze_sample(d)
    r_val = validate_sample(d)
    sg, s_notes = structural_grade(d, v, r_val)

    sem: Optional[Dict[str, Any]] = None
    if semantic_check_sample and src_root and src_root.is_dir():
        try:
            _br = bin_root.expanduser().resolve() if bin_root else None
            sem = semantic_check_sample(
                d,
                src_root,
                high_w=high_w,
                viol_detail_limit=5,
                asm_grounded=asm_grounded,
                bin_root=_br,
            )
        except Exception as e:
            sem = {"skip": True, "reason": str(e)}
    elif not src_root:
        sem = None
    else:
        sem = {"skip": True, "reason": "no src_root"}

    mg, m_notes = semantic_grade(sem)
    n_bb = v.get("n_blocks", 0)
    aligned_bb = sum(
        1
        for i in range(n_bb)
        if i < len(d.get("alignment_matrix") or [])
        and any(float(x) > 1e-9 for x in (d.get("alignment_matrix") or [])[i])
    )

    out: Dict[str, Any] = {
        "func_name": d.get("func_name"),
        "opt_level": d.get("opt_level"),
        "n_bb": n_bb,
        "n_bb_matrix_nonempty": aligned_bb,
        "structural": sg,
        "structural_notes": "; ".join(s_notes) if s_notes else "",
        "semantic": mg,
        "semantic_notes": "; ".join(m_notes) if m_notes else "",
        "viol_peak": sem.get("n_viol_peak") if sem and not sem.get("skip") else "-",
        "viol_highw": sem.get("n_viol_highw") if sem and not sem.get("skip") else "-",
        "n_bb_checked_sem": sem.get("n_bb_checked") if sem and not sem.get("skip") else "-",
        "semantic_skip_reason": (sem.get("reason") or "") if sem and sem.get("skip") else "",
        "matrix_lines_consistent": v.get("matrix_lines_consistent"),
        "coverage": r_val.get("coverage"),
        "order_consistency": v.get("order_consistency"),
    }
    if sem and sem.get("asm_grounded") and not sem.get("skip"):
        out["sem_asm_grounded"] = True
        out["asm_cfg_ok"] = sem.get("asm_cfg_ok")
        out["n_viol_dwarf_json"] = sem.get("n_viol_dwarf_json")
        out["n_viol_no_pc_line"] = sem.get("n_viol_no_pc_line")
    else:
        out["sem_asm_grounded"] = False
    return out


def expand_to_bb_rows(
    d: Dict[str, Any],
    dataset: str,
    file_s: str,
    src_root: Optional[Path],
    high_w: float,
    *,
    asm_grounded: bool = False,
    bin_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """每条基本块一行：结构（是否有行/M 行）+ 语义（是否进入 check_sample 的 BB 级主峰/高W 检查）。"""
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    n_bb = len(bl)
    v = analyze_sample(d)
    bb_pri: List[int] = list(v.get("bb_primary") or [])

    fskip = ""
    sem: Optional[Dict[str, Any]] = None
    if not semantic_check_sample:
        fskip = "no check_alignment_semantics"
    elif not src_root or not src_root.is_dir():
        fskip = "no src_root"
    else:
        try:
            _br = bin_root.expanduser().resolve() if bin_root else None
            sem = semantic_check_sample(
                d,
                src_root,
                high_w=high_w,
                viol_detail_limit=None,
                asm_grounded=asm_grounded,
                bin_root=_br,
            )
        except Exception as e:
            sem = {"skip": True, "reason": str(e)}
        if sem and sem.get("skip"):
            fskip = str(sem.get("reason", ""))

    viol_peak_bbs: set = set()
    highw_by_bb: Dict[int, int] = {}
    viol_dwarf_json_bbs: set = set()
    bb_pc_line_count: List[int] = []
    asm_cfg_ok_f: Optional[bool] = None
    if sem and not sem.get("skip"):
        for x in sem.get("viol_peak") or []:
            viol_peak_bbs.add(int(x["bb"]))
        for x in sem.get("viol_highw") or []:
            b = int(x["bb"])
            highw_by_bb[b] = highw_by_bb.get(b, 0) + 1
        for x in sem.get("viol_dwarf_json") or []:
            viol_dwarf_json_bbs.add(int(x["bb"]))
        bb_pc_line_count = list(sem.get("bb_pc_line_count") or [])
        if sem.get("asm_grounded"):
            asm_cfg_ok_f = bool(sem.get("asm_cfg_ok"))

    rows: List[Dict[str, Any]] = []
    for i in range(n_bb):
        has_dw = bool(bl[i])
        row_m = M[i] if i < len(M) else []
        matrix_nonempty = bool(row_m) and max((float(x) for x in row_m), default=0) > 1e-9
        j_peak = bb_pri[i] if i < len(bb_pri) else -1
        max_m = max((float(x) for x in row_m), default=0.0)

        if has_dw and not matrix_nonempty:
            struct_bb = "FAIL"
        elif not has_dw and matrix_nonempty:
            struct_bb = "WARN"
        else:
            struct_bb = "OK"

        order_note = ""
        if i > 0 and i - 1 < len(bb_pri) and i < len(bb_pri):
            a, b2 = bb_pri[i - 1], bb_pri[i]
            if a >= 0 and b2 >= 0 and b2 < a - 2:
                order_note = f"stmt_jump_{a}_to_{b2}"

        row = M[i] if i < len(M) else []
        D = set(bl[i] or [])
        sem_checked = (
            bool(D) and bool(row) and max((float(x) for x in row), default=0) > 1e-6
        )

        if fskip:
            sem_peak = "SKIP"
            sem_highw = "SKIP"
        elif not sem_checked:
            sem_peak = "NA"
            sem_highw = "NA"
        else:
            sem_peak = "WARN" if i in viol_peak_bbs else "OK"
            sem_highw = "WARN" if i in highw_by_bb else "OK"

        pc_n = bb_pc_line_count[i] if i < len(bb_pc_line_count) else ""
        if asm_grounded:
            if fskip or not (sem and not sem.get("skip") and sem.get("asm_grounded")):
                djp = "NA"
            elif i in viol_dwarf_json_bbs:
                djp = "WARN"
            elif not bb_pc_line_count:
                djp = "NA"
            else:
                djp = "OK"
        else:
            djp = ""

        row_d: Dict[str, Any] = {
            "dataset": dataset,
            "file": file_s,
            "func_name": d.get("func_name"),
            "opt_level": d.get("opt_level"),
            "bb_idx": i,
            "has_dwarf_lines": has_dw,
            "dwarf_line_count": len(D),
            "matrix_nonempty": matrix_nonempty,
            "struct_bb": struct_bb,
            "order_issue": order_note,
            "j_peak": j_peak,
            "M_peak": round(max_m, 6),
            "sem_bb_in_check": sem_checked,
            "sem_peak": sem_peak,
            "sem_highw": sem_highw,
            "highw_viol_nj": highw_by_bb.get(i, 0) if sem_checked else 0,
            "func_sem_skip_reason": fskip,
        }
        if asm_grounded:
            row_d["sem_asm_grounded"] = True
            row_d["pc_dwarf_line_n"] = pc_n
            row_d["dwarf_json_vs_pc"] = djp
            row_d["func_asm_cfg_ok"] = (
                asm_cfg_ok_f if asm_cfg_ok_f is not None else ""
            )
        else:
            row_d["sem_asm_grounded"] = False
        rows.append(row_d)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="BB 对齐：结构 + 语义双监测")
    ap.add_argument("jsonl", nargs="?", type=Path, help="单个 jsonl（与 --scan 二选一）")
    ap.add_argument("--src-root", type=Path, help="源码根（语义监测必需）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理条数（0=全部）")
    ap.add_argument("--high-w", type=float, default=0.85)
    ap.add_argument("--csv", action="store_true", help="输出 CSV 到 stdout")
    ap.add_argument("--scan", action="store_true", help="扫描 9 项目 × 多档 jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--search-dir", action="append", default=[])
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--opt-levels", default="0,1,2,3,os")
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS) if DEFAULT_PROJECTS else "zlib")
    ap.add_argument("--per-file", type=int, default=0, help="每个 jsonl 顺序读前 N 条（0=全部；与随机二选一）")
    ap.add_argument(
        "--random-per-file",
        type=int,
        default=0,
        help="每个 jsonl **随机**抽 N 条（蓄水池，与 --per-file 互斥时优先随机）",
    )
    ap.add_argument("--seed", type=int, default=42, help="随机种子（--random-per-file 时生效）")
    ap.add_argument(
        "--by-bb",
        action="store_true",
        help="输出按基本块展开（每 BB 一行）；默认按函数一行",
    )
    ap.add_argument(
        "--src-map",
        action="append",
        default=[],
        metavar="PROJ=PATH",
        help="项目名=源码根，语义用",
    )
    ap.add_argument(
        "--sem-asm-grounded",
        action="store_true",
        help="语义：ELF+Capstone 重切 BB + .debug_line 按 PC 重放行号，再核对主峰/高W（需 binary 可读，路径不对时用 --bin-root）",
    )
    ap.add_argument(
        "--bin-root",
        type=Path,
        default=None,
        help="json 中 binary 路径失效时，在此目录按文件名/rglob 查找 ELF",
    )
    args = ap.parse_args()

    src_map: Dict[str, Path] = {}
    for p in args.src_map:
        if "=" in p:
            k, v = p.split("=", 1)
            src_map[k.strip()] = Path(v).expanduser().resolve()

    preset = {
        "zlib": "/mnt/e/structlift/zlib_build",
        "curl": "/mnt/e/structlift_src_datasets/curl/curl-8.7.1",
        "sqlite": "/mnt/e/structlift_src_datasets/sqlite/sqlite-autoconf-3520000",
        "busybox": "/mnt/e/structlift_src_datasets/busybox/busybox-1.36.1",
        "openssl": "/mnt/e/structlift_src_datasets/openssl/openssl-3.3.1",
        "openssh": "/mnt/e/structlift_src_datasets/openssh/openssh-9.8p1",
        "libxml2": "/mnt/e/structlift_src_datasets/libxml2/libxml2-2.12.7",
        "ffmpeg": "/mnt/e/structlift_src_datasets/ffmpeg/ffmpeg-7.1",
        "coreutils": "/mnt/e/structlift_src_datasets/coreutils/coreutils-9.4",
    }
    for k, v in preset.items():
        pp = Path(v)
        if pp.is_dir():
            src_map.setdefault(k, pp.resolve())

    def iter_jsonl_paths() -> List[Tuple[str, str, Path]]:
        out: List[Tuple[str, str, Path]] = []
        if not args.scan or not locate_jsonl:
            return out
        out_dir = args.out_dir.expanduser().resolve()
        if args.all_default_paths and default_multi_root_search_list:
            search_dirs = default_multi_root_search_list(out_dir)
            for p in args.search_dir:
                search_dirs.extend([Path(p).expanduser().resolve()])
            search_dirs = dedupe_paths(search_dirs) if dedupe_paths else search_dirs
        else:
            search_dirs = [out_dir] + [Path(p).expanduser().resolve() for p in args.search_dir]
        opts = parse_opt_levels(args.opt_levels)
        for proj in [x.strip() for x in args.projects.split(",") if x.strip()]:
            for ot in opts:
                name = binskel_jsonl_name(proj, ot)
                pu, _ = locate_jsonl(name, search_dirs)
                if pu:
                    label = f"{proj}_{ot}"
                    out.append((proj, label, pu))
        return out

    results: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)

    _bin_root = args.bin_root.expanduser().resolve() if args.bin_root else None

    def append_one(
        d: Dict[str, Any], label: str, file_s: str, root: Optional[Path]
    ) -> None:
        if args.by_bb:
            results.extend(
                expand_to_bb_rows(
                    d,
                    label,
                    file_s,
                    root,
                    args.high_w,
                    asm_grounded=args.sem_asm_grounded,
                    bin_root=_bin_root,
                )
            )
        else:
            r = process_one(
                d,
                root,
                args.high_w,
                asm_grounded=args.sem_asm_grounded,
                bin_root=_bin_root,
            )
            r["file"] = file_s
            r["dataset"] = label
            results.append(r)

    if args.scan:
        paths = iter_jsonl_paths()
        if not paths:
            print("未找到任何 jsonl，检查 --out-dir / --all-default-paths", file=sys.stderr)
            sys.exit(1)
        use_random = args.random_per_file > 0
        for proj, label, jp in paths:
            src_root = src_map.get(proj)
            if use_random:
                dicts = reservoir_sample_dicts(jp, args.random_per_file, rng)
                for d in dicts:
                    append_one(d, label, str(jp), src_root)
                    if args.limit and len(results) >= args.limit:
                        break
            else:
                n_read = 0
                with open(jp, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if args.per_file and n_read >= args.per_file:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        append_one(d, label, str(jp), src_root)
                        n_read += 1
                        if args.limit and len(results) >= args.limit:
                            break
            if args.limit and len(results) >= args.limit:
                break
    else:
        if not args.jsonl:
            ap.error("请指定 jsonl 或使用 --scan")
        jp = args.jsonl.expanduser()
        if args.random_per_file > 0:
            dicts = reservoir_sample_dicts(jp, args.random_per_file, rng)
            for d in dicts:
                append_one(d, jp.name, str(jp), args.src_root)
                if args.limit and len(results) >= args.limit:
                    break
        else:
            n_read = 0
            with open(jp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if args.per_file and n_read >= args.per_file:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    append_one(d, jp.name, str(jp), args.src_root)
                    n_read += 1
                    if args.limit and len(results) >= args.limit:
                        break

    if args.csv:
        default_cols = (
            [
                "dataset",
                "file",
                "func_name",
                "opt_level",
                "bb_idx",
                "struct_bb",
                "sem_peak",
                "sem_highw",
                "sem_asm_grounded",
                "pc_dwarf_line_n",
                "dwarf_json_vs_pc",
                "func_asm_cfg_ok",
            ]
            if args.by_bb
            else ["dataset"]
        )
        w = csv.DictWriter(
            sys.stdout,
            fieldnames=list(results[0].keys()) if results else default_cols,
            extrasaction="ignore",
        )
        if results:
            w.writeheader()
            w.writerows(results)
        return

    if args.by_bb:
        n = len(results)
        print(
            "=== BB 对齐：结构 + 语义（**基本块粒度**，启发式，非金标准证明）===\n"
        )
        print(f"输出行数（每 BB 一行）: {n}")
        if n:
            st_ok = sum(1 for r in results if r.get("struct_bb") == "OK")
            st_warn = sum(1 for r in results if r.get("struct_bb") == "WARN")
            st_fail = sum(1 for r in results if r.get("struct_bb") == "FAIL")
            sp_ok = sum(1 for r in results if r.get("sem_peak") == "OK")
            sp_warn = sum(1 for r in results if r.get("sem_peak") == "WARN")
            sp_na = sum(1 for r in results if r.get("sem_peak") == "NA")
            sp_skip = sum(1 for r in results if r.get("sem_peak") == "SKIP")
            sh_warn = sum(1 for r in results if r.get("sem_highw") == "WARN")
            in_chk = sum(1 for r in results if r.get("sem_bb_in_check"))
            print(
                f"结构(每 BB): OK={st_ok}  WARN={st_warn}  FAIL={st_fail}  "
                f"(M 与 DWARF 行、矩阵非空)"
            )
            print(
                f"语义主峰(每 BB): OK={sp_ok}  WARN={sp_warn}  NA={sp_na}  "
                f"SKIP={sp_skip}  （sem_bb_in_check 计数={in_chk}）"
            )
            print(f"语义高 W（每 BB 行）: WARN 行数={sh_warn}")
        print("\n--- 前 20 行摘要 ---")
        for r in results[:20]:
            fn = (r.get("func_name") or "?")[:40]
            print(
                f"{r.get('dataset', '?')} | {fn} | bb={r.get('bb_idx')} | "
                f"结构={r.get('struct_bb')} | sem_peak={r.get('sem_peak')} | "
                f"sem_highw={r.get('sem_highw')}"
            )
        if n > 20:
            print(f"... 共 {n} 行，用 --csv 导出全表")
        return

    # summary（函数粒度）
    n = len(results)
    st_ok = sum(1 for r in results if r["structural"] == "OK")
    st_warn = sum(1 for r in results if r["structural"] == "WARN")
    st_fail = sum(1 for r in results if r["structural"] == "FAIL")
    se_ok = sum(1 for r in results if r["semantic"] == "OK")
    se_warn = sum(1 for r in results if r["semantic"] == "WARN")
    se_skip = sum(1 for r in results if r["semantic"] == "SKIP")
    tot_bb = sum(int(r.get("n_bb") or 0) for r in results)
    viol_pk = sum(
        int(r["viol_peak"]) for r in results if isinstance(r.get("viol_peak"), int)
    )

    print("=== BB 对齐：结构监测 + 语义监测（启发式，非金标准证明）===\n")
    print(f"样本条数: {n}")
    if n:
        print(
            f"结构: OK={st_ok}  WARN={st_warn}  FAIL={st_fail}  "
            f"(M与block_src_lines自洽、空BB、顺序、覆盖率阈值)"
        )
        print(
            f"语义: OK={se_ok}  WARN={se_warn}  SKIP={se_skip}  "
            f"(主峰/高W vs tree-sitter语句与DWARF)"
        )
        print(f"总 BB 数(累加): {tot_bb}")
        print(f"语义 viol_peak 累加(仅整数列): {viol_pk}")
    print("\n--- 前 20 条摘要 ---")
    for r in results[:20]:
        print(
            f"{r.get('dataset','?')} | {r.get('func_name','?')[:40]} | "
            f"结构={r['structural']} | 语义={r['semantic']} | "
            f"BB={r.get('n_bb')} | viol_peak={r.get('viol_peak')}"
        )
    if n > 20:
        print(f"... 共 {n} 条，用 --csv 导出全表")


if __name__ == "__main__":
    main()
