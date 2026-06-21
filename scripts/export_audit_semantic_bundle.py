#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为固定审计样本导出「源码片段 +（可选）ELF 反汇编」供人工/LLM 做语义对齐判断。

jsonl 内 bin_opcodes 为 mnemonic 的 hash，不可逆；若提供可访问的 ELF（与构建 jsonl 时相同），
则按符号表重切 BB 并打印 capstone 文本，与 bin_block_lengths 对齐检查。

**路径说明**：jsonl 里路径常为 ``.../datasets/zlib/.../zlib_o0``。请把 ``--elf-dir`` 设为
**datasets 目录本身**（其下应有 ``zlib/zlib-1.3.1/zlib_o0``）。脚本会取 ``/datasets/`` 后的尾路径拼接，
不依赖 ``--path-subst`` 是否把 /home 整段映射对。

用法（WSL，推荐）:
  python3 scripts/export_audit_semantic_bundle.py \\
    --datasets ~/structlift_datasets \\
    --elf-dir /mnt/e/struclift_wsl/datasets \\
    --path-subst /home/wuqiongmin/struclift_wsl=/mnt/e/struclift_wsl

用法（仅有 Windows 盘 jsonl + 本地 zlib 源码树）:
  python scripts/export_audit_semantic_bundle.py \\
    --datasets E:/structlift_datasets \\
    --source-alias gzwrite.c=E:/structlift/zlib_build/gzwrite.c \\
    -o audit_bundle.md

可重复指定 --source-alias basename=path 或 --source-dir DIR（按文件名在 DIR 下递归查找，慎用）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_validate import analyze_sample  # noqa: E402

SAMPLES: list[tuple[str, int]] = []
for idx in [110, 135, 50, 119, 95]:
    SAMPLES.append(("binskel_zlib_o0.jsonl", idx))
for idx in [105, 266, 185, 186, 209]:
    SAMPLES.append(("binskel_curl_o0.jsonl", idx))
for idx in [1609, 1834, 495, 425, 2356]:
    SAMPLES.append(("binskel_sqlite_o0.jsonl", idx))
for idx in [2528, 4057, 2023, 561, 2574]:
    SAMPLES.append(("binskel_busybox_o0.jsonl", idx))
for idx in [10632, 7832, 7292, 864, 1382]:
    SAMPLES.append(("binskel_openssl_o0.jsonl", idx))
for idx in [6, 58, 60, 52, 47]:
    SAMPLES.append(("binskel_libxml2_o0.jsonl", idx))
for idx in [840, 311, 12646, 12645, 7061]:
    SAMPLES.append(("binskel_ffmpeg_o0.jsonl", idx))


def load_by_line_idx(path: Path, target: int) -> dict | None:
    idx = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            idx += 1
            if idx == target:
                return json.loads(line)
    return None


def apply_subst(p: str, substs: list[tuple[str, str]]) -> str:
    s = p
    for a, b in substs:
        if s.startswith(a):
            s = b + s[len(a) :]
            break
    return s


def _tail_after_datasets_marker(path: str) -> str | None:
    """从任意前缀的绝对路径取出 datasets/ 之后的相对路径（用于挂载根不同）。"""
    p = path.replace("\\", "/")
    for marker in ("/datasets/", "/data_sets/"):
        k = p.find(marker)
        if k >= 0:
            return p[k + len(marker) :].lstrip("/")
    return None


def _dedupe_roots(roots: list[Path | None]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        if r is None or not r.is_dir():
            continue
        k = str(r.resolve())
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def resolve_source_path(
    raw: str | None,
    substs: list[tuple[str, str]],
    aliases: dict[str, Path],
    source_dir: Path | None,
    artifact_roots: list[Path],
) -> tuple[Path | None, str]:
    """返回 (路径, 简短说明) 便于调试。"""
    if not raw:
        return None, "无 source_file"
    s = apply_subst(raw, substs)
    cand = Path(s)
    if cand.is_file():
        return cand, f"subst直链存在: {cand}"
    base = Path(raw).name
    if base in aliases and aliases[base].is_file():
        return aliases[base], f"--source-alias: {aliases[base]}"

    tail = _tail_after_datasets_marker(raw)
    if tail:
        for root in artifact_roots:
            q = root / tail
            if q.is_file():
                return q, f"尾路径命中: {root}/{tail}"

    if source_dir and source_dir.is_dir():
        for p in source_dir.rglob(base):
            if p.is_file():
                return p, f"--source-dir rglob: {p}"
    roots_s = ",".join(str(x) for x in artifact_roots) or "(无artifact根)"
    return None, f"未找到 (subst后={s}, tail={tail!r}, roots=[{roots_s}])"


def resolve_elf_path(
    raw: str | None, substs: list[tuple[str, str]], artifact_roots: list[Path]
) -> tuple[Path | None, str]:
    if not raw:
        return None, "无 binary 字段"
    s = apply_subst(raw, substs)
    p = Path(s)
    if p.is_file():
        return p, f"subst直链: {p}"

    tail = _tail_after_datasets_marker(raw)
    if tail:
        for root in artifact_roots:
            q = root / tail
            if q.is_file():
                return q, f"尾路径命中: {root}/{tail}"

    bn = p.name
    all_hits: list[Path] = []
    for root in artifact_roots:
        all_hits.extend(x for x in root.rglob(bn) if x.is_file())
    if len(all_hits) == 1:
        return all_hits[0], f"唯一 rglob({bn}): {all_hits[0]}"
    if len(all_hits) > 1:
        raw_n = raw.replace("\\", "/")
        best = None
        best_score = -1
        for h in all_hits:
            hs = str(h).replace("\\", "/")
            score = sum(1 for part in raw_n.split("/") if part and part in hs)
            if score > best_score:
                best_score = score
                best = h
        if best:
            return best, f"多命中取最像: {best} (共{len(all_hits)}个)"
    roots_s = ",".join(str(x) for x in artifact_roots) or "(无artifact根)"
    return None, f"未找到 (subst后={s}, tail={tail!r}, roots=[{roots_s}])"


def disasm_function_blocks(elf_path: Path, func_name: str) -> tuple[list[list[str]], str] | tuple[None, str]:
    try:
        from elftools.elf.elffile import ELFFile
        from build_binskel_dataset_md import build_cfg_for_function, iter_functions
    except ImportError as e:
        return None, f"import 失败: {e}"

    try:
        with elf_path.open("rb") as f:
            elffile = ELFFile(f)
            text_sec = elffile.get_section_by_name(".text")
            if text_sec is None:
                return None, "无 .text"
            text_bytes = text_sec.data()
            base_addr = text_sec["sh_addr"]
            addr = size = None
            for name, a, sz in iter_functions(elffile):
                if name == func_name:
                    addr, size = a, sz
                    break
            if addr is None or size is None:
                return None, f"符号表中无函数 {func_name}"
            start = addr - base_addr
            if start < 0 or start + size > len(text_bytes):
                return None, "函数范围越界"
            code_bytes = text_bytes[start : start + size]
            cfg, _ = build_cfg_for_function(code_bytes, addr, func_name)
            out: list[list[str]] = []
            for bb in cfg.blocks:
                lines = []
                for inst in bb.instructions:
                    lines.append(getattr(inst, "raw_text", "") or "?")
                out.append(lines)
        return out, "ok"
    except Exception as e:
        return None, str(e)


def snippet_for_lines(file_lines: list[str], lo: int, hi: int) -> str:
    """1-based inclusive line numbers."""
    parts = []
    for ln in range(lo, hi + 1):
        if 1 <= ln <= len(file_lines):
            parts.append(f"L{ln:5d}| {file_lines[ln - 1].rstrip()}")
    return "\n".join(parts) if parts else "(行号超出文件)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument(
        "--path-subst",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="路径前缀替换，可多次",
    )
    ap.add_argument(
        "--elf-dir",
        type=Path,
        default=None,
        help="datasets 根目录（内含 zlib/、curl/… 子树；用于 datasets/ 尾路径拼接与 rglob）",
    )
    ap.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="已弃用别名：请用 --artifact-root；仍兼容为第一个 artifact 根",
    )
    ap.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="DIR",
        help="构建产物根目录，可重复。须能拼上 json 里 /datasets/ 后的路径。"
        "例: .../datasets 与 .../datasets/ffmpeg_binskel",
    )
    ap.add_argument("--source-dir", type=Path, default=None, help="在目录下按源文件名递归找")
    ap.add_argument(
        "--source-alias",
        action="append",
        default=[],
        metavar="file.c=PATH",
        help="精确指定某文件名对应的绝对路径",
    )
    ap.add_argument("--max-bb", type=int, default=24, help="每条样本最多打印多少个 BB")
    ap.add_argument(
        "--only-jsonl",
        type=str,
        default="",
        help="仅处理该文件名（如 binskel_zlib_o0.jsonl），加速大库",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    substs: list[tuple[str, str]] = []
    for s in args.path_subst:
        if "=" not in s:
            continue
        a, b = s.split("=", 1)
        substs.append((a, b))

    aliases: dict[str, Path] = {}
    for s in args.source_alias:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        aliases[k.strip()] = Path(v.strip())

    root = args.datasets.expanduser().resolve()
    elf_dir = args.elf_dir.expanduser().resolve() if args.elf_dir else None
    legacy = args.datasets_root.expanduser().resolve() if args.datasets_root else None
    extra = [Path(p).expanduser().resolve() for p in args.artifact_root]
    artifact_roots = _dedupe_roots([elf_dir, legacy] + extra)
    source_dir = (
        args.source_dir.expanduser().resolve() if args.source_dir else None
    )
    out_lines: list[str] = []

    def emit(x: str = "") -> None:
        out_lines.append(x)

    emit("# 审计样本：源码片段 + 反汇编（若 ELF 可用）")
    emit()

    todo = SAMPLES
    if args.only_jsonl:
        todo = [(n, i) for n, i in SAMPLES if n == args.only_jsonl]

    for name, li in todo:
        path = root / name
        emit("---")
        emit(f"## {name} line_idx={li}")
        if not path.is_file():
            emit("**错误**: jsonl 不存在")
            emit()
            continue
        d = load_by_line_idx(path, li)
        if not d:
            emit("**错误**: 行索引不存在")
            emit()
            continue

        fn = d.get("func_name", "?")
        sf_raw = d.get("source_file")
        bin_raw = d.get("binary")
        emit(f"**函数**: `{fn}`")
        emit(f"**记录中的 binary**: `{bin_raw}`")
        emit(f"**记录中的 source_file**: `{sf_raw}`")

        ar = analyze_sample(d)
        M = d.get("alignment_matrix") or []
        block_lines = d.get("block_src_lines") or []
        source_lines = d.get("source_lines") or []
        bb_primary: list[int] = ar.get("bb_primary") or []
        bin_bl = d.get("bin_block_lengths") or []

        sp, src_how = resolve_source_path(
            sf_raw, substs, aliases, source_dir, artifact_roots
        )
        emit(f"SOURCE_RESOLVE: {src_how}")
        if sp:
            emit(f"**解析到的源码路径**: `{sp}`")
            try:
                src_file_lines = sp.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as e:
                src_file_lines = []
                emit(f"**源码读取失败**: {e}")
        else:
            src_file_lines = []
            emit("**源码**: 未找到本地文件（请用 --path-subst / --source-alias / --source-dir / --elf-dir 指向含 datasets 子树的根）")

        ep, elf_how = resolve_elf_path(bin_raw, substs, artifact_roots)
        emit(f"ELF_RESOLVE: {elf_how}")
        asm_blocks: list[list[str]] | None = None
        asm_note = ""
        if ep:
            asm_blocks, asm_note = disasm_function_blocks(ep, fn)
            emit(f"DISASM_META: elf_ok=1 note={asm_note!r}")
            emit(f"**解析到的 ELF**: `{ep}` → 反汇编: {asm_note}")
        else:
            emit(f"DISASM_META: elf_ok=0 note={elf_how[:200]!r}")
            emit("**ELF**: 未找到（无法打印助记符汇编；仅 json 内为 opcode hash）")

        if asm_blocks and bin_bl:
            if len(asm_blocks) != len(bin_bl):
                emit(
                    f"**警告**: 重切 BB 数 {len(asm_blocks)} ≠ jsonl bin_block_lengths {len(bin_bl)} "
                    "（可能不同 capstone/构建版本，反汇编仅作参考）"
                )
            else:
                for bi in range(len(asm_blocks)):
                    if bi < len(bin_bl) and len(asm_blocks[bi]) != bin_bl[bi]:
                        emit(
                            f"**警告**: BB{bi} 指令数 反汇编={len(asm_blocks[bi])} jsonl={bin_bl[bi]}"
                        )
                        break

        n_show = min(args.max_bb, len(block_lines), len(bb_primary) or len(block_lines))
        emit()
        emit("### 逐 BB（主对齐列 + DWARF 行 + 源码 + 汇编）")
        for bi in range(n_show):
            j = bb_primary[bi] if bi < len(bb_primary) else -1
            bl = block_lines[bi] if bi < len(block_lines) else []
            stmt_info = ""
            src_snip = ""
            if j >= 0 and j < len(source_lines):
                ln = source_lines[j]
                stmt_info = f"stmt_idx={j} → L{ln}"
                if src_file_lines and isinstance(ln, (int, float)):
                    src_snip = snippet_for_lines(src_file_lines, int(ln), int(ln))
            elif j < 0:
                stmt_info = "stmt_idx=(空)"
            else:
                stmt_info = f"stmt_idx={j} (越界)"

            asm_txt = ""
            if asm_blocks and bi < len(asm_blocks):
                asm_txt = " | ".join(asm_blocks[bi][:12])
                if len(asm_blocks[bi]) > 12:
                    asm_txt += " ..."

            emit(f"- **BB{bi}** | DWARF行 {bl} | {stmt_info}")
            if src_snip:
                emit("  ```c")
                for sl in src_snip.split("\n")[:8]:
                    emit("  " + sl)
                emit("  ```")
            if asm_txt:
                emit(f"ASM_BB{bi}: {asm_txt}")
                emit(f"  `asm: {asm_txt}`")
            emit("")

        emit(f"*矩阵自洽 matrix_vs_block_lines*: {'PASS' if ar['matrix_lines_consistent'] else 'FAIL'}")
        emit()

    text = "\n".join(out_lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"已写入 {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
