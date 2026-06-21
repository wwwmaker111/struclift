import sys
sys.path.insert(0, "scripts")
from audit_four_tier_alignment_45x3 import iter_filtered_line_indices
from pathlib import Path
p = Path("/mnt/e/structlift_datasets/binskel_ffmpeg_o0.jsonl")
c = iter_filtered_line_indices(p, 8, 0.65)
print("candidates min_bb=8 mean_conf>=0.65:", len(c))
c2 = iter_filtered_line_indices(p, 8, None)
print("candidates min_bb=8 no conf filter:", len(c2))
