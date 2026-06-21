#!/usr/bin/env python3
"""一次性统计 binskel_zlib_o0.jsonl（重建后复核用）。"""
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "binskel_zlib_o0.jsonl")
    tiers = Counter()
    n_bb = []
    cov = []
    opt = Counter()
    with p.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            tiers[o.get("src_encoding_tier", "?")] += 1
            opt[str(o.get("opt_level", "?"))] += 1
            n_bb.append(len(o.get("bin_block_lengths") or []))
            c = o.get("bb_alignment_coverage")
            if c is not None:
                cov.append(float(c))
    n = len(n_bb)
    print(f"文件: {p.resolve()}")
    print(f"样本数: {n}")
    print(f"src_encoding_tier: {dict(tiers)}")
    print(f"opt_level: {dict(opt)}")
    if n_bb:
        s = sorted(n_bb)
        print(
            f"每样本 BB 数 len(bin_block_lengths): min={min(n_bb)} median={s[n // 2]} max={max(n_bb)} sum={sum(n_bb)}"
        )
    if cov:
        sc = sorted(cov)
        print(
            f"bb_alignment_coverage: min={min(cov):.4f} median={sc[n // 2]:.4f} max={max(cov):.4f}"
        )


if __name__ == "__main__":
    main()
