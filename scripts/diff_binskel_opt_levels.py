#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定性：多档 binskel jsonl 是「逐样本真相同」还是「只有汇总看起来像」。

按 (func_name, source_file) 对齐样本，对比与优化强相关的字段指纹；与 validate 的小数位无关。

用法:
  python3 scripts/diff_binskel_opt_levels.py \\
      ~/datasets/ffmpeg_binskel/binskel_ffmpeg_o0.jsonl \\
      ~/datasets/ffmpeg_binskel/binskel_ffmpeg_o3.jsonl

  python3 scripts/diff_binskel_opt_levels.py o0.jsonl o1.jsonl o2.jsonl o3.jsonl

耗时粗估（四文件各 ~300MB、数万～十万级函数）:
  家用 SSD + ext4: 常约 3～10 分钟；WSL 读 /mnt/e 或机械盘可能 15～30+ 分钟。
  内存峰值约「基准档 + 当前对比档」两套 dict，建议可用内存 ≥ 4～8GB。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _key(rec: dict) -> Tuple[str, str]:
    fn = str(rec.get("func_name") or "")
    sf = str(rec.get("source_file") or "")
    return (fn, sf)


def _fingerprint(rec: dict) -> str:
    """与 -O 等级强相关的子集（不含 opt_level 字段本身）。"""
    payload = {
        "bin_block_lengths": rec.get("bin_block_lengths"),
        "bin_edge_index": rec.get("bin_edge_index"),
        "bin_edge_type": rec.get("bin_edge_type"),
        "bin_opcodes": rec.get("bin_opcodes"),
        "block_src_lines": rec.get("block_src_lines"),
        "alignment_confidence": rec.get("alignment_confidence"),
        "alignment_matrix": rec.get("alignment_matrix"),
        "binary": rec.get("binary"),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_index(path: Path) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = _key(d)
            out[k] = d
    return out


def compare_pair(
    path_a: Path,
    path_b: Path,
    max_examples: int,
    index_a: Dict[Tuple[str, str], dict] | None = None,
) -> None:
    ia = index_a if index_a is not None else load_index(path_a)
    ib = load_index(path_b)
    ka, kb = set(ia), set(ib)
    inter = ka & kb
    only_a = ka - kb
    only_b = kb - ka

    same_fp = 0
    diff_fp = 0
    examples: List[str] = []

    for k in inter:
        fa = _fingerprint(ia[k])
        fb = _fingerprint(ib[k])
        if fa == fb:
            same_fp += 1
        else:
            diff_fp += 1
            if len(examples) < max_examples:
                fn, sf = k
                examples.append(f"  func_name={fn!r} source_file={sf!r}")

    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print(f"  A 样本数: {len(ia)}  B 样本数: {len(ib)}")
    print(f"  键交集 (同 func+文件): {len(inter)}")
    print(f"  仅 A 有: {len(only_a)}  仅 B 有: {len(only_b)}")
    print(f"  交集内指纹相同: {same_fp}  指纹不同: {diff_fp}")
    if inter:
        pct = 100.0 * diff_fp / len(inter)
        print(f"  => 交集内「与优化相关字段」不同的比例: {pct:.2f}%")
    if examples:
        print("  指纹不同示例 (最多 {} 条):".format(max_examples))
        for e in examples:
            print(e)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="对比两档或多档 binskel jsonl 是否逐样本真相同")
    ap.add_argument(
        "jsonl",
        nargs="+",
        type=Path,
        help="至少两个 jsonl；若多于两个，则依次与第一个文件对比",
    )
    ap.add_argument(
        "--examples",
        type=int,
        default=5,
        help="每对打印几条「指纹不同」的 func 名",
    )
    args = ap.parse_args()
    paths = [p.expanduser().resolve() for p in args.jsonl]
    for p in paths:
        if not p.is_file():
            print(f"错误: 不是文件: {p}", file=sys.stderr)
            sys.exit(1)

    base = paths[0]
    # 只读一遍基准档，避免 o0 被重复解析 3 次（大 jsonl 可省约 1/3 时间）
    index_base = load_index(base)
    for other in paths[1:]:
        compare_pair(base, other, args.examples, index_base)

    print(
        "解读: 若「指纹不同」>0 且比例很高 → 数据并非四档相同，之前更像统计/平均/小数位问题。\n"
        "若交集内指纹几乎全相同 → 再查是否误用同一 ELF 或旧 jsonl 拷贝。"
    )


if __name__ == "__main__":
    main()
