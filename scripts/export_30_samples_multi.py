#!/usr/bin/env python3
"""从 zlib/sqlite/busybox/openssl 四项目各抽若干样本，共 30 个，导出供 pattern_labels 核查"""
import json
import subprocess
import sys
from pathlib import Path

# PatternType: 0=SEQUENTIAL 1=IF_HEADER 2=IF_THEN_BODY 3=IF_ELSE_BODY
# 4=LOOP_HEADER 5=LOOP_BODY 6=LOOP_EXIT 7=SWITCH_HEADER 8=SWITCH_CASE_BODY 9=FUNC_PROLOGUE_EPILOGUE
PATTERN_NAMES = {
    0: "SEQUENTIAL", 1: "IF_HEADER", 2: "IF_THEN_BODY", 3: "IF_ELSE_BODY",
    4: "LOOP_HEADER", 5: "LOOP_BODY", 6: "LOOP_EXIT",
    7: "SWITCH_HEADER", 8: "SWITCH_CASE_BODY", 9: "FUNC_PROLOGUE_EPILOGUE",
}

datasets = [
    ("zlib", Path.home() / "structlift_datasets/binskel_zlib_o0.jsonl", 8),
    ("sqlite", Path.home() / "structlift_datasets/binskel_sqlite_o0.jsonl", 8),
    ("busybox", Path.home() / "structlift_datasets/binskel_busybox_o0.jsonl", 7),
    ("openssl", Path.home() / "structlift_datasets/binskel_openssl_o0.jsonl", 7),
]

selected = []
for proj, p, k in datasets:
    p = p.expanduser()
    if not p.exists():
        continue
    with open(p) as f:
        n = sum(1 for _ in f)
    if n == 0:
        continue
    want = set(min(int(j * n / (k + 1)), n - 1) for j in range(k))
    max_idx = max(want) if want else 0
    with open(p) as f:
        for idx, line in enumerate(f):
            if idx in want:
                d = json.loads(line)
                selected.append((proj, idx, d.get("func_name", "?"), d.get("pattern_labels", [])))
            if idx >= max_idx:
                break

selected = selected[:30]

out_file = Path(__file__).parent.parent / "pattern_30_samples_check.txt"
py = Path(__file__).parent / "export_sample_for_manual_check.py"

with open(out_file, "w", encoding="utf-8") as out:
    out.write("# PatternType 共 10 种: 0 SEQUENTIAL 1 IF_HEADER 2 IF_THEN_BODY 3 IF_ELSE_BODY\n")
    out.write("# 4 LOOP_HEADER 5 LOOP_BODY 6 LOOP_EXIT 7 SWITCH_HEADER 8 SWITCH_CASE_BODY 9 FUNC_PROLOGUE_EPILOGUE\n\n")
    for i, (proj, line_idx, fn, pl) in enumerate(selected):
        pl_str = ",".join(PATTERN_NAMES.get(x, str(x)) for x in pl[:25])
        if len(pl) > 25:
            pl_str += f"...({len(pl)}块)"
        out.write(f"\n{'='*80}\n>>> 样本 {i+1}/30: [{proj}] {fn} (idx={line_idx})\n")
        out.write(f"    pattern_labels: {pl_str}\n{'='*80}\n")
        jsonl = next(p.expanduser() for pname, p, _ in datasets if pname == proj and p.expanduser().exists())
        r = subprocess.run(
            [sys.executable, str(py), str(jsonl), "--index", str(line_idx)],
            capture_output=True, text=True, timeout=90, cwd=Path(__file__).parent.parent
        )
        out.write(r.stdout)
        if r.stderr:
            out.write("\n[stderr]\n" + r.stderr)

print(f"已导出 30 个样本到 {out_file}")
print("样本:", [(s[0], s[2][:30], f"{len(s[3])}块") for s in selected])
