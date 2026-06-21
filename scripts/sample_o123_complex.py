#!/usr/bin/env python3
import json
from pathlib import Path

BASE = Path("/home/wuqiongmin/structlift_datasets")
PROJECTS = ["zlib", "sqlite", "busybox", "openssl", "coreutils", "curl"]


def calc(d):
    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    bl = d.get("block_src_lines") or []
    nb = len(d.get("bin_block_lengths") or bl)
    ns = len(M[0]) if M and M[0] else 0
    used = sum(1 for j in range(ns) if any((row[j] if j < len(row) else 0) > 0 for row in M)) if ns else 0
    bb_best = [max(range(len(row)), key=lambda j: row[j]) for row in M if row and max(row) > 0]
    collapse = max((bb_best.count(c) / len(bb_best) for c in set(bb_best)), default=0.0)
    nz = [w for row in W for w in row if w > 0]
    meanw = (sum(nz) / len(nz)) if nz else 0.0
    high = (sum(1 for w in nz if w > 0.7) / len(nz)) if nz else 0.0
    src_lines = sorted({ln for ls in bl for ln in ls})
    span = (max(src_lines) - min(src_lines) + 1) if src_lines else 0
    sig = d.get("signal_stats") or {}
    both = sig.get("n_both", 0)
    conflict = sig.get("n_conflict", 0)
    conflict_rate = (conflict / both) if both else 0.0
    return nb, ns, used, collapse, meanw, high, span, conflict_rate


for opt in ["1", "2", "3"]:
    cand = []
    for proj in PROJECTS:
        p = BASE / f"binskel_{proj}_o{opt}.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                d = json.loads(line)
                nb, ns, used, collapse, meanw, high, span, cr = calc(d)
                if nb < 8 or used < 3:
                    continue
                score = nb * 0.5 + meanw * 20 + high * 10 + used * 1.5 - collapse * 15 - cr * 10
                cand.append((score, proj, idx, d.get("func_name", "?"), nb, used, ns, collapse, meanw, high, span, cr))
    cand.sort(reverse=True, key=lambda x: x[0])
    picked = []
    seen = set()
    for it in cand:
        key = (it[1], it[3])
        if key in seen:
            continue
        seen.add(key)
        picked.append(it)
        if len(picked) >= 5:
            break

    print(f"\n=== O{opt} 复杂样本Top5 ===")
    for i, it in enumerate(picked, 1):
        _, proj, idx, fn, nb, used, ns, collapse, meanw, high, span, cr = it
        flags = []
        if collapse >= 0.8:
            flags.append("COLLAPSE")
        if cr >= 0.3:
            flags.append("DWARF_IR_CONFLICT_HIGH")
        if meanw < 0.7:
            flags.append("LOW_MEANW")
        if not flags:
            flags = ["LIKELY_OK"]
        print(
            f"#{i} {proj}::{fn} line={idx+1} bb={nb} used={used}/{ns} meanW={meanw:.3f} "
            f"high={high:.3f} collapse={collapse:.2f} span={span} conflict_rate={cr:.2f} => {','.join(flags)}"
        )
