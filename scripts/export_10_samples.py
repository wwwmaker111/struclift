#!/usr/bin/env python3
"""导出 10 个样本到文件，用于 pattern_labels 核查"""
import json
import subprocess
import sys
from pathlib import Path

jsonl = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "structlift_datasets/binskel_zlib_o0.jsonl"
out_file = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path("zlib_10_samples_check.txt")

funcs = []
with open(jsonl) as f:
    for line in f:
        d = json.loads(line)
        funcs.append(d.get("func_name", "?"))

# 均匀抽取 + 指定关键样本（含 if/else/return 的典型函数）
n = len(funcs)
indices = [0, n//10, n//5, n//3, n//2, 2*n//3, 3*n//4, n-5, n-3, n-1]
extra_names = ["gz_reset", "gztell64", "inflateResetKeep", "deflate_fast", "gzclose", "gzopen"]
name2idx = {fn: i for i, fn in enumerate(funcs) if fn}

selected = []
seen = set()
for i in indices:
    fn = funcs[i]
    if fn and fn not in seen:
        seen.add(fn)
        selected.append((i, fn))
for fn in extra_names:
    if fn in name2idx and fn not in seen:
        selected.append((name2idx[fn], fn))
        seen.add(fn)

selected = selected[:18]  # 多取几个
py = Path(__file__).parent / "export_sample_for_manual_check.py"
with open(out_file, "w", encoding="utf-8") as out:
    for idx, (line_idx, fn) in enumerate(selected):
        out.write(f"\n{'='*80}\n>>> 样本 {idx+1}/{len(selected)}: {fn} (行 {line_idx+1})\n{'='*80}\n")
        r = subprocess.run(
            [sys.executable, str(py), str(jsonl), "--index", str(line_idx)],
            capture_output=True, text=True, timeout=60, cwd=Path(__file__).parent.parent
        )
        out.write(r.stdout)
        if r.stderr:
            out.write("\n[stderr]\n" + r.stderr)

print(f"已导出 {len(selected)} 个样本到 {out_file}")
print("函数:", [fn for _, fn in selected])
