#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="分析 high_confidence_audit 报告对应样本的复杂度/告警")
    ap.add_argument("--audit", default="/home/wuqiongmin/high_confidence_audit.txt")
    ap.add_argument("--data-dir", default="/home/wuqiongmin/structlift_datasets")
    args = ap.parse_args()

    audit_path = Path(args.audit)
    base = Path(args.data_dir)
    txt = audit_path.read_text(encoding="utf-8", errors="replace")

    pat = re.compile(
        r"--- \[(\w+)\] #(\d+)/(\d+)\s+jsonl行号=(\d+)\(0-based idx=(\d+)\)\s+func=([^\n]+) ---\n\n"
        r"  metrics: mean_conf=([0-9.]+)  high_W_ratio=([0-9.]+)  low_W_ratio=([0-9.]+)  "
        r"BB_cov=([0-9.]+)  order=([0-9.]+)  cf=([0-9.]+)",
        re.S,
    )

    rows = []
    for m in pat.finditer(txt):
        proj, _, _, _, idx, fn, mean, high, low, cov, order, cf = m.groups()
        idx = int(idx)
        p = base / f"binskel_{proj}_o0.jsonl"
        d = None
        with p.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == idx:
                    d = json.loads(line)
                    break
        if d is None:
            continue

        M = d.get("alignment_matrix") or []
        W = d.get("alignment_confidence") or []
        bl = d.get("block_src_lines") or []
        nb = len(d.get("bin_block_lengths") or bl)
        ns = len(M[0]) if M and M[0] else 0
        used = (
            sum(
                1
                for j in range(ns)
                if any((row[j] if j < len(row) else 0) > 0 for row in M)
            )
            if ns
            else 0
        )
        bb_best = []
        for row in M:
            if row and max(row) > 0:
                bb_best.append(max(range(len(row)), key=lambda j: row[j]))
        collapse = (
            max((bb_best.count(c) / len(bb_best) for c in set(bb_best)))
            if bb_best
            else 0.0
        )
        src_lines = sorted({ln for ls in bl for ln in ls})
        span = (max(src_lines) - min(src_lines) + 1) if src_lines else 0
        uniq = len(src_lines)

        flags = []
        if nb <= 1:
            flags.append("TRIVIAL")
        if used <= 1:
            flags.append("ONE_COL")
        if collapse >= 0.80:
            flags.append("COLLAPSE")
        if float(order) < 0.80 or float(cf) < 0.80:
            flags.append("STRUCT_WARN")

        rows.append(
            {
                "proj": proj,
                "fn": fn.strip(),
                "idx": idx,
                "mean": float(mean),
                "high": float(high),
                "low": float(low),
                "cov": float(cov),
                "order": float(order),
                "cf": float(cf),
                "blocks": nb,
                "cols": ns,
                "used_cols": used,
                "collapse": collapse,
                "src_span": span,
                "src_unique": uniq,
                "flags": flags,
            }
        )

    print("TOTAL", len(rows))
    for r in rows:
        flag_txt = ";".join(r["flags"]) if r["flags"] else "OK"
        print(
            f"{r['proj']:9s} | {r['fn'][:34]:34s} | bb={r['blocks']:3d} "
            f"col={r['used_cols']:2d}/{r['cols']:2d} | collapse={r['collapse']:.2f} "
            f"| span={r['src_span']:3d} uniq={r['src_unique']:3d} "
            f"| order={r['order']:.2f} cf={r['cf']:.2f} | {flag_txt}"
        )

    agg = defaultdict(list)
    for r in rows:
        agg[r["proj"]].append(r)

    print("\nPER_PROJECT")
    for p, arr in agg.items():
        n = len(arr)
        trivial = sum(1 for x in arr if "TRIVIAL" in x["flags"])
        one_col = sum(1 for x in arr if "ONE_COL" in x["flags"])
        collapse = sum(1 for x in arr if "COLLAPSE" in x["flags"])
        struct = sum(1 for x in arr if "STRUCT_WARN" in x["flags"])
        avg_blocks = sum(x["blocks"] for x in arr) / n
        print(
            f"{p:9s}: n={n} trivial={trivial} one_col={one_col} "
            f"collapse_warn={collapse} struct_warn={struct} avg_blocks={avg_blocks:.1f}"
        )


if __name__ == "__main__":
    main()
