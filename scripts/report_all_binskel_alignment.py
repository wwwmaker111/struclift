#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键生成「全部项目」binskel 对齐总表（默认 8 项目 × O0–O3，含 openssh）。

在多个常见根目录中自动查找 jsonl（与 binskel_alignment_batch_stats.py --all-default-paths 相同）。

用法:
  cd /path/to/structlift && python3 scripts/report_all_binskel_alignment.py
  python3 scripts/report_all_binskel_alignment.py --tsv
  python3 scripts/report_all_binskel_alignment.py --projects zlib,curl,ffmpeg
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def main() -> None:
    argv = [sys.argv[0], "--all-default-paths"] + sys.argv[1:]
    sys.argv = argv
    from binskel_alignment_batch_stats import main as batch_main

    batch_main()


if __name__ == "__main__":
    main()
