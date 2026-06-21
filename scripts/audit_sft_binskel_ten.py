#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计「十份」binskel SFT jsonl：源码可读性 + Qwen tokenizer 下 ``sft_input_ids`` 合法性。

对 train/val × o0–os 共 10 个文件逐一统计（默认 ``{prefix}_{split}_{tag}_sft.jsonl``）。

**源码侧**（与 ``augment_binskel_sft`` / ``audit_binskel_sft_source_parse`` 一致）：

- 先按 ``--rewrite-prefix FROM TO``（可多次）改写 ``source_file`` 字符串再解析路径；
- 再试绝对路径 + 各 ``--source-root`` 相对拼接；
- ``_line_to_text`` 至少一行 → **可读源码行**；
- augment 等价 weak：``lt`` 空且 ``any(block_src_lines)``.

**tokenizer 侧**：

- ``pip install transformers``，``--tokenizer`` 建议与训练/增广目录一致（**含 FILL 占位符的那份** ``save_pretrained`` 目录，
  否则 ``len(tokenizer)`` 可能与 jsonl 中 id 范围不一致）。
- 检查 ``sft_input_ids``：非空、元素为 ``int``、且对每个 id 满足 ``0 <= id < len(tokenizer)``；
  （``sft_labels`` 中的 ``-100`` 若写入 json **不在**此项检查范围内，我们只查 ``sft_input_ids``。）

输出：每个文件的行数、「源码可读」「仅弱默认」「路径不可解析」「词表合法」以及 **同时具备**「可读源码 + id 合法」的条数与占比。

示例::

  python3 scripts/audit_sft_binskel_ten.py \\
    --base-dir /data/chaoni/WQM/datasets \\
    --name-prefix CD \\
    --tokenizer /data/chaoni/WQM/datasets/struclift_tokenizer_qwen3.6_27b_fill \\
    --source-root /data/chaoni/WQM/source_datasets \\
    --rewrite-prefix /mnt/e/structlift_src_datasets /data/chaoni/WQM/source_datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAGS = ("o0", "o1", "o2", "o3", "os")
SPLITS = ("train", "val")


def _rewrite(sf: str, pairs: List[Tuple[str, str]]) -> str:
    if not sf or not pairs:
        return sf
    x = sf.replace("\\", "/")
    for frm, to in pairs:
        ff = str(frm).replace("\\", "/").rstrip("/")
        tt = str(to).replace("\\", "/").rstrip("/")
        if ff and x.startswith(ff):
            x = tt + x[len(ff) :]
    return x


def _line_to_text(source_path: Path) -> Dict[int, str]:
    if not source_path.is_file():
        return {}
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return {i: line for i, line in enumerate(text.splitlines(), 1)}


def _resolve(sf_raw: str, roots: List[Path]) -> Optional[Path]:
    if not sf_raw.strip():
        return None
    p = Path(str(sf_raw))
    if p.is_file():
        return p.resolve()
    for root in roots:
        q = (root / p).resolve()
        if q.is_file():
            return q
        q2 = (root / p.name).resolve()
        if q2.is_file():
            return q2
    return None


def _weak_like(record: Dict[str, Any], lt: Dict[int, str]) -> bool:
    if lt:
        return False
    bl = record.get("block_src_lines")
    return bool(bl and isinstance(bl, list) and any(bl))


def _source_bucket(
    record: Dict[str, Any],
    roots: List[Path],
    rewrites: List[Tuple[str, str]],
) -> str:
    """readable | weak_fallback | no_src | unresolvable | empty_file."""
    sf0 = str(record.get("source_file") or "").strip()
    if not sf0:
        return "no_src"
    sf = _rewrite(sf0, rewrites)
    rec = dict(record)
    rec["source_file"] = sf
    sp = _resolve(sf, roots)
    if sp is None:
        return "unresolvable"
    lt = _line_to_text(sp)
    if lt:
        return "readable"
    if _weak_like(record, lt):
        return "weak_fallback"
    try:
        if sp.stat().st_size == 0:
            return "empty_file"
    except OSError:
        pass
    return "unreadable_nonempty"


def _sft_ids_ok(sid: Any, vocab: int) -> Tuple[bool, str]:
    if not isinstance(sid, list) or len(sid) == 0:
        return False, "empty_or_missing"
    for i, x in enumerate(sid):
        if not isinstance(x, int):
            return False, f"non_int_at_{i}"
        if x < 0 or x >= vocab:
            return False, f"oob_at_{i}_id_{x}_max_{vocab - 1}"
    return True, ""


def audit_file(
    path: Path,
    *,
    vocab: int,
    roots: List[Path],
    rewrites: List[Tuple[str, str]],
    progress_every: int,
) -> Dict[str, Any]:
    n = 0
    src_readable = 0
    src_weak = 0
    src_no = 0
    src_unres = 0
    src_empty = 0
    src_other = 0
    sft_ok = 0
    sft_bad = 0
    both_ok = 0

    if progress_every > 0:
        logger.info("%s: 开始逐行扫描（每 %d 行一条进度）…", path.name, progress_every)
    else:
        logger.info("%s: 开始逐行扫描（未启用进度日志）…", path.name)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n += 1
            d = json.loads(s)
            sb = _source_bucket(d, roots, rewrites)
            if sb == "readable":
                src_readable += 1
            elif sb == "weak_fallback":
                src_weak += 1
            elif sb == "no_src":
                src_no += 1
            elif sb == "unresolvable":
                src_unres += 1
            elif sb == "empty_file":
                src_empty += 1
            else:
                src_other += 1

            ok, _why = _sft_ids_ok(d.get("sft_input_ids"), vocab)
            if ok:
                sft_ok += 1
            else:
                sft_bad += 1

            if ok and sb == "readable":
                both_ok += 1

            if progress_every > 0 and n % progress_every == 0:
                logger.info("  %s: 已扫 %d 行 …", path.name, n)

    return {
        "path": path,
        "n": n,
        "src_readable": src_readable,
        "src_weak": src_weak,
        "src_no_src": src_no,
        "src_unresolvable": src_unres,
        "src_empty_file": src_empty,
        "src_other": src_other,
        "sft_vocab_ok": sft_ok,
        "sft_vocab_bad": sft_bad,
        "both_readable_and_sft_ok": both_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="审计十份 SFT jsonl：源码可读 + tokenizer id 合法")
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--name-prefix", type=str, default="CD", help="如 CD → CD_train_o0_sft.jsonl")
    ap.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HF 名或本地目录（建议用 augment 保存的含 FILL 的目录）",
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        action="append",
        dest="source_roots",
        help="可多次，与 augment 一致",
    )
    ap.add_argument(
        "--rewrite-prefix",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        dest="pairs",
        help="改写 source_file 前缀后再解析（可多次）",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25000,
        metavar="N",
        help="每 N 条打 INFO（默认 25000；0 关闭）",
    )
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--tags", nargs="+", default=list(TAGS))
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("需要: pip install transformers")
        sys.exit(1)

    base = Path(args.base_dir).expanduser().resolve()
    roots = [Path(p).expanduser().resolve() for p in (args.source_roots or [])]
    pairs = [(str(a), str(b)) for a, b in (args.pairs or [])]

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab = len(tok)
    logger.info(
        "tokenizer=%s → len(tokenizer)=%d（用此上限检查 sft_input_ids）",
        args.tokenizer,
        vocab,
    )

    paths: List[Path] = []
    for sp in args.splits:
        for tg in args.tags:
            p = base / f"{args.name_prefix}_{sp}_{tg}_sft.jsonl"
            if not p.is_file():
                logger.error("缺失: %s", p)
                raise SystemExit(1)
            paths.append(p)

    print(
        f"[audit_sft_binskel_ten] 将顺序审计 {len(paths)} 个文件（无 "
        f"--progress-every 间隔内可能较久才有下一条 INFO）",
        flush=True,
    )

    grand = defaultdict(int)

    lines: List[str] = []
    W = lines.append
    W("=" * 80)
    W(f"源码根 source_roots={[str(r) for r in roots]}")
    W(f"rewrite_prefixes={pairs!r}")
    W("")

    for p in paths:
        logger.info("==== %s ====", p.name)
        r = audit_file(p, vocab=vocab, roots=roots, rewrites=pairs, progress_every=int(args.progress_every))
        n = max(r["n"], 1)
        W(p.name)
        W(f"  总行数: {r['n']}")
        pct = lambda x: f"{100.0 * x / n:.2f}%"
        W(f"  源码可读 readable:           {r['src_readable']:8d}  ({pct(r['src_readable'])})")
        W(f"  augment级弱(default+block):  {r['src_weak']:8d}  ({pct(r['src_weak'])})")
        W(f"  无 source_file 字段:        {r['src_no_src']:8d}")
        W(f"  路径不可解析 unresolvable:  {r['src_unresolvable']:8d}  ({pct(r['src_unresolvable'])})")
        W(f"  源文件空:                   {r['src_empty_file']:8d}")
        W(f"  其它源码状态:               {r['src_other']:8d}")
        W(f"  sft_input_ids 词表内合法:   {r['sft_vocab_ok']:8d}  ({pct(r['sft_vocab_ok'])})")
        W(f"  sft_input_ids 缺/越界:     {r['sft_vocab_bad']:8d}  ({pct(r['sft_vocab_bad'])})")
        W(f"  同时满足「可读源码+id合法»: {r['both_readable_and_sft_ok']:8d}  ({pct(r['both_readable_and_sft_ok'])})")
        W("")

        for k in (
            "n",
            "src_readable",
            "src_weak",
            "src_no_src",
            "src_unresolvable",
            "src_empty_file",
            "src_other",
            "sft_vocab_ok",
            "sft_vocab_bad",
            "both_readable_and_sft_ok",
        ):
            grand[k] += r[k]

    tot = grand["n"]
    W("=" * 80)
    W(f"十个文件合计 总行数={tot}")
    if tot > 0:
        def pctg(x):
            return f"{100.0 * x / tot:.2f}%"

        W(f"合计 源码可读:                {grand['src_readable']:8d}  ({pctg(grand['src_readable'])})")
        W(f"合计 unresolvable:            {grand['src_unresolvable']:8d}  ({pctg(grand['src_unresolvable'])})")
        W(f"合计 sft id 合法:             {grand['sft_vocab_ok']:8d}  ({pctg(grand['sft_vocab_ok'])})")
        W(f"合计 可读源码且 sft 合法:     {grand['both_readable_and_sft_ok']:8d}  ({pctg(grand['both_readable_and_sft_ok'])})")
    W("")
    W("说明：")
    W("  · 「可读源码」≈ augment 时能读到磁盘 .c 行图；_weak_ 仍有 sft_id 多半是 block 弱默认.")
    W("  · 「sft id 合法」取决于 --tokenizer：请用 augment 时同一套 vocab（否则大量 oob）.")
    report = "\n".join(lines) + "\n"
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()
