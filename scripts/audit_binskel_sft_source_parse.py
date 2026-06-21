#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 binskel jsonl 中 Stage3 SFT 与「是否从磁盘源码读行」的关系。

与 ``augment_binskel_sft.py`` 一致：

- ``_resolve_source_path``：``source_file`` 绝对路径可走通，或相对于 ``--source-root``。
- ``_line_to_text``：能读出至少一行源文件正文 → ``lt`` 非空。
- augment 中会打 weak 的条件（同上脚本 L299–305）：「``lt`` 为空 **且**
  ``block_src_lines`` 按 ``any(block_src_lines)`` 为非空」（与上游列表_truthiness 一致）。

输出统计（每条样本归入互斥的一类）::

  readable_lines   lt 非空 — 能从源码文件解析出行文本，可作为「强源码行 GT」的前提
  augment_weak_row lt 空且 block_src_lines truthy — 与 augment 中「GT 弱默认」触发条件一致
  no_src_path      记录无 ``source_file`` 或为空
  unresolvable     有 ``source_file`` 但无法在磁盘解析出路径（请检查挂载与 ``--source-root``）
  empty_src_file   路径存在但读不到任何行（空文件）
  no_sft           缺少 ``sft_input_ids`` 或为空列表

用法（Qwen 十文件 train+val × 五档）::

  python scripts/audit_binskel_sft_source_parse.py \\\\
    --base-dir /data/chaoni/WQM/datasets \\\\
    --name-prefix Qwen3.6_C \\\\
    --splits train val

若只统计 full 不按 _p20::

  （默认 suffix 为空）

若只看 _p20 抽样结果::

  --suffix _p20

可多次传入 ``--source-root`` 与 augment 一致。
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAGS_DEFAULT = ("o0", "o1", "o2", "o3", "os")


def _line_to_text(source_path: Path) -> Dict[int, str]:
    if not source_path.is_file():
        return {}
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: Dict[int, str] = {}
    for i, line in enumerate(text.splitlines(), 1):
        out[i] = line
    return out


def _resolve_source_path(
    record: Dict[str, Any], source_roots: List[Optional[Path]]
) -> Optional[Path]:
    raw = record.get("source_file")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p.resolve()
    for source_root in source_roots:
        if source_root is None:
            continue
        q = (Path(source_root) / p).resolve()
        if q.is_file():
            return q
        q2 = (Path(source_root) / p.name).resolve()
        if q2.is_file():
            return q2
    return None


def _weak_branch_like_augment(record: Dict[str, Any], lt: Dict[int, str]) -> bool:
    """与 augment_binskel_sft.process_record 中 debug 分支同条件。"""
    if lt:
        return False
    bl = record.get("block_src_lines")
    return bool(bl and isinstance(bl, list) and any(bl))


def _classify(record: Dict[str, Any], source_roots: List[Optional[Path]]) -> Tuple[str, str]:
    """返回 (类别, 附注说明)。"""
    sid = record.get("sft_input_ids")
    if not isinstance(sid, list) or len(sid) == 0:
        return "no_sft", ""

    raw = record.get("source_file")
    if raw is None or str(raw).strip() == "":
        return "no_src_path", ""

    src_resolved = _resolve_source_path(record, source_roots)
    if src_resolved is None:
        return "unresolvable", str(raw)[:120]

    lt = _line_to_text(src_resolved)
    if lt:
        return "readable_lines", ""

    if _weak_branch_like_augment(record, lt):
        return "augment_weak_row", ""

    try:
        st = src_resolved.stat().st_size
    except OSError:
        st = -1
    if st == 0:
        return "empty_src_file", ""

    # 有路径、读文本失败或空（非 0 字节则可能编码全坏 → 无线）
    return "unreadable_nonempty", str(src_resolved)


def audit_file(path: Path, source_roots: List[Optional[Path]], progress_every: int) -> Dict[str, Any]:
    counts: Dict[str, int] = defaultdict(int)
    n_line = 0

    with path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            n_line += 1
            try:
                d = json.loads(s)
            except json.JSONDecodeError as e:
                counts["json_error"] += 1
                logger.warning("%s:%d JSON 解析失败: %s", path.name, li, e)
                continue

            cat, _note = _classify(d, source_roots)
            counts[cat] += 1

            if progress_every > 0 and n_line % progress_every == 0:
                logger.info("  %s: 已读 %d 行 …", path.name, n_line)

    counts["_physical_lines_nonempty"] = n_line
    return {"path": path, "counts": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser(description="审计 SFT 与源码读盘是否一致（对齐 augment_binskel_sft）")
    ap.add_argument("--base-dir", type=Path, help="数据集目录（与 --preset 合用）")
    ap.add_argument("--name-prefix", type=str, default="Qwen3.6_C", help="如 Qwen3.6_C")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help='要扫描的 split 名（默认 train val）',
    )
    ap.add_argument(
        "--tags",
        nargs="+",
        default=list(TAGS_DEFAULT),
        help="优化档标签，默认 o0 o1 o2 o3 os",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help='文件名中 _o0 之前的额外后缀（如 _p20 → ..._train_o0_p20.jsonl）',
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        dest="jsonl_paths",
        help="可显式指定若干 jsonl（可重复）；若给此项则忽略 --base-dir 拼装",
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        action="append",
        dest="source_roots",
        help=" augment 同源：解析相对 source_file 时尝试的根目录（可多次）",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=15000,
        metavar="N",
        help="每 N 条非空行打一条进度（0 关闭）",
    )
    args = ap.parse_args()

    roots: List[Optional[Path]] = []
    if args.source_roots:
        roots.extend([Path(p).expanduser().resolve() for p in args.source_roots])
    roots.append(None)

    paths: List[Path] = []
    if args.jsonl_paths:
        paths = [Path(p).expanduser().resolve() for p in args.jsonl_paths]
    else:
        if not args.base_dir:
            logger.error("请提供 --jsonl … 或同时提供 --base-dir")
            raise SystemExit(1)
        base = Path(args.base_dir).expanduser().resolve()
        suf = str(args.suffix)
        for sp in args.splits:
            for tg in args.tags:
                fname = f"{args.name_prefix}_{sp}_{tg}{suf}.jsonl"
                p = base / fname
                if not p.is_file():
                    logger.error("缺失: %s", p)
                    raise SystemExit(1)
                paths.append(p)

    logger.info("将审计 %d 个文件 source_roots=%s", len(paths), roots[:-1])

    grand: Dict[str, int] = defaultdict(int)
    reports: List[Dict[str, Any]] = []

    for p in paths:
        logger.info("==== %s ====", p)
        r = audit_file(p, roots, int(args.progress_every))
        reports.append(r)
        for k, v in r["counts"].items():
            if not k.startswith("_"):
                grand[k] += v
        gpos = r["counts"].get("_physical_lines_nonempty", 0)
        logger.info("  非空行: %d", gpos)
        for key in (
            "readable_lines",
            "augment_weak_row",
            "no_src_path",
            "unresolvable",
            "empty_src_file",
            "unreadable_nonempty",
            "no_sft",
            "json_error",
        ):
            if r["counts"].get(key):
                logger.info("  %-20s %8d", key + ":", r["counts"][key])

    print()
    print("=" * 72)
    print("汇总（所有列出文件累计）")
    print("=" * 72)
    total_nonempty = sum(re["counts"].get("_physical_lines_nonempty", 0) for re in reports)
    print("非空物理行合计:", total_nonempty)
    for key in (
        "readable_lines",
        "augment_weak_row",
        "no_src_path",
        "unresolvable",
        "empty_src_file",
        "unreadable_nonempty",
        "no_sft",
        "json_error",
    ):
        if grand.get(key):
            pct = 100.0 * grand[key] / max(total_nonempty, 1)
            print(f"  {key:22s} {grand[key]:8d}  ({pct:5.2f}%)")

    print()
    print("说明：")
    print("  readable_lines     = 与 augment 一致：source 解析成功且至少读到一行源文件正文。")
    print("  augment_weak_row   = lt 空且 any(block_src_lines)（augment 会走「弱默认」debug 场景）。")
    print("  unresolvable       = 有 source_file 字符串但 --source-root 仍找不到文件。")
    print("  （若全为 unresolvable，请在服务器上加上实际源码挂载路径再跑 --source-root。）")


if __name__ == "__main__":
    main()
