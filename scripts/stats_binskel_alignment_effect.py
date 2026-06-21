#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计当前 binskel 对齐效果（默认 8 项目 × O0–O3，多目录自动找 jsonl）。

在 validate_alignment 指标之外，增加：
  bb_cov 均%、stmt/行档占比、M_gt↔block_src_lines 自洽率（与 audit_two_samples 同源）。

用法:
  cd /path/to/structlift && python3 scripts/stats_binskel_alignment_effect.py

  python3 scripts/stats_binskel_alignment_effect.py --out-dir /mnt/e/structlift_datasets
  python3 scripts/stats_binskel_alignment_effect.py --out-dir ~/structlift_datasets \\
      --search-dir ~/struclift_wsl/datasets/ffmpeg-7.1/../ffmpeg_binskel

  python3 scripts/stats_binskel_alignment_effect.py --tsv   # 制表符，便于贴 Excel

与下列命令等价（本脚本自动加 --all-default-paths）:
  python3 scripts/binskel_alignment_batch_stats.py --all-default-paths
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if "--help" not in sys.argv and "-h" not in sys.argv:
    if "--all-default-paths" not in sys.argv:
        sys.argv = [sys.argv[0], "--all-default-paths"] + sys.argv[1:]

from binskel_alignment_batch_stats import main  # noqa: E402

if __name__ == "__main__":
    main()
