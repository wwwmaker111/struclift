"""分析 18 样本中 j* 语句粒度：行范围 span vs 函数体 span。"""
import re

lines = open(r"e:\structlift\samples_9proj_o0_18_disasm.txt", encoding="utf-8", errors="replace").readlines()

cur_proj = cur_func = ""
cur_nbb = 0
cur_bb = -1
results = []

for line in lines:
    m = re.match(r"^# PROJECT:\s+(\S+)", line)
    if m:
        cur_proj = m.group(1)
        continue
    m = re.match(r"^# func_name:\s+(.+)$", line)
    if m:
        cur_func = m.group(1).strip()
        continue
    m = re.match(r"^# n_bb:\s+(\d+)", line)
    if m:
        cur_nbb = int(m.group(1))
        continue
    m = re.match(r"^## BB (\d+)", line)
    if m:
        cur_bb = int(m.group(1))
        continue
    m = re.match(r"^.+ts_type=(\S+)\s+ast_type_id=\S+\s+\(([^)]+)\)", line)
    if m and cur_bb >= 0:
        ts_type = m.group(1)
        ast_label = m.group(2)
        results.append({
            "proj": cur_proj, "func": cur_func, "n_bb": cur_nbb,
            "bb": cur_bb, "ts_type": ts_type, "ast_label": ast_label,
            "span": 0, "lo": 0, "hi": 0,
        })
        continue
    m = re.match(r"^.+L(\d+)\D+L(\d+)", line)
    if m and results and results[-1]["span"] == 0:
        lo, hi = int(m.group(1)), int(m.group(2))
        results[-1]["lo"] = lo
        results[-1]["hi"] = hi
        results[-1]["span"] = hi - lo + 1
        continue

print(f"{'proj':<12} {'func':<35} {'BB':>4} {'n_bb':>5} {'ts_type':<25} {'ast_label':<20} {'range':>20} {'span':>6}  grade")
print("-" * 135)

fine = coarse = medium = 0
for r in results:
    sp = r["span"]
    if sp <= 10:
        g = "FINE"
        fine += 1
    elif sp <= 40:
        g = "MED"
        medium += 1
    else:
        g = "COARSE"
        coarse += 1
    rng = f"L{r['lo']}-L{r['hi']}"
    print(f"{r['proj']:<12} {r['func']:<35} {r['bb']:>4} {r['n_bb']:>5} {r['ts_type']:<25} {r['ast_label']:<20} {rng:>20} {sp:>6}  {g}")

total = fine + medium + coarse
print(f"\nTotal {total} BBs:")
print(f"  FINE   (<=10 lines): {fine:>3} ({100*fine/total:.0f}%)")
print(f"  MED  (11-40 lines):  {medium:>3} ({100*medium/total:.0f}%)")
print(f"  COARSE  (>40 lines): {coarse:>3} ({100*coarse/total:.0f}%)")
