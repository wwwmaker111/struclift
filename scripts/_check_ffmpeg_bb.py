import json
from pathlib import Path
p = Path("/mnt/e/structlift_datasets/binskel_ffmpeg_o0.jsonl")
if not p.is_file():
    print("NOT FOUND"); exit()
counts = []
with open(p) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        bl = d.get("block_src_lines") or []
        counts.append(len(bl))
print(f"total samples: {len(counts)}")
print(f"max bb: {max(counts) if counts else 0}")
print(f">=8: {sum(1 for c in counts if c>=8)}")
print(f">=5: {sum(1 for c in counts if c>=5)}")
print(f">=3: {sum(1 for c in counts if c>=3)}")
