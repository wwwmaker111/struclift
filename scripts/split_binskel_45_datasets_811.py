#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 9 项目 × 5 优化档（O0–O3 + Os）共 45 个 binskel jsonl 各自按 8:1:1 划分为 train / val / test。

命名约定（与仓库 scripts/build_binskel_9proj_5opt.sh / multibin 脚本一致）：
  - zlib, curl, sqlite, busybox, openssl, libxml2:
      binskel_<proj>_o0.jsonl … o3.jsonl、os.jsonl
  - openssh, ffmpeg, coreutils（multibin）:
      binskel_<proj>_o0_multibin.jsonl …、os_multibin.jsonl

划分规则（每文件独立、按行打乱）：
  n_train = (8 * n) // 10
  n_val   = (n - n_train) // 2
  n_test  = n - n_train - n_val
（保证 n_train + n_val + n_test = n；极小 n 时 val/test 可能为 0。）

用法（WSL）::
  python3 scripts/split_binskel_45_datasets_811.py \\
      --in-dir /mnt/e/structlift_datasets \\
      --out-dir /mnt/e/structlift_datasets/split_811_seed42 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPTS = ("o0", "o1", "o2", "o3", "os")

# 6 项目：单 ELF 命名
PROJECTS_SIMPLE = (
    "zlib",
    "curl",
    "sqlite",
    "busybox",
    "openssl",
    "libxml2",
)
# 3 项目：multibin 合并 jsonl
PROJECTS_MULTIBIN = ("openssh", "ffmpeg", "coreutils")


def expected_jsonl_paths(in_dir: Path) -> list[tuple[str, Path]]:
    """返回 (逻辑名, 路径) 列表，逻辑名用于 manifest，如 zlib_o0。"""
    out: list[tuple[str, Path]] = []
    for proj in PROJECTS_SIMPLE:
        for opt in OPTS:
            name = f"binskel_{proj}_{opt}.jsonl"
            out.append((f"{proj}_{opt}", in_dir / name))
    for proj in PROJECTS_MULTIBIN:
        for opt in OPTS:
            name = f"binskel_{proj}_{opt}_multibin.jsonl"
            out.append((f"{proj}_{opt}_multibin", in_dir / name))
    return out


def split_counts(n: int) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = (8 * n) // 10
    rem = n - n_train
    n_val = rem // 2
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def split_lines(lines: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    n_train, n_val, n_test = split_counts(len(lines))
    tr = [lines[i] for i in order[:n_train]]
    va = [lines[i] for i in order[n_train : n_train + n_val]]
    te = [lines[i] for i in order[n_train + n_val :]]
    assert len(tr) + len(va) + len(te) == len(lines)
    return tr, va, te


def main() -> None:
    ap = argparse.ArgumentParser(description="45 个 binskel jsonl → 8:1:1 train/val/test")
    ap.add_argument("--in-dir", type=Path, required=True, help="原始 jsonl 所在目录")
    ap.add_argument("--out-dir", type=Path, required=True, help="输出根目录（下建 train/ val/ test/）")
    ap.add_argument("--seed", type=int, default=42, help="每文件内打乱随机种子")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计不写文件")
    args = ap.parse_args()

    in_dir = args.in_dir.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    if not args.dry_run:
        for sub in ("train", "val", "test"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[str] = []
    manifest_rows.append("logical_name\tfilename\texists\tn_lines\tn_train\tn_val\tn_test")

    missing = 0
    done = 0

    for logical, path in expected_jsonl_paths(in_dir):
        stem = path.name
        if not path.is_file():
            logger.warning("缺失: %s", path)
            manifest_rows.append(f"{logical}\t{stem}\t0\t0\t0\t0\t0")
            missing += 1
            continue

        raw_lines: list[str] = []
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_lines.append(line)

        n = len(raw_lines)
        n_train, n_val, n_test = split_counts(n)
        # 每文件用不同子种子，避免所有文件同一顺序
        sub_seed = args.seed + hash(logical) % 100000
        tr, va, te = split_lines(raw_lines, sub_seed)

        manifest_rows.append(
            f"{logical}\t{stem}\t1\t{n}\t{len(tr)}\t{len(va)}\t{len(te)}",
        )

        if not args.dry_run:
            for split_name, chunk in (("train", tr), ("val", va), ("test", te)):
                out_path = out_root / split_name / stem
                with out_path.open("w", encoding="utf-8") as wf:
                    for ln in chunk:
                        wf.write(ln + "\n")
        done += 1
        logger.info("%s: n=%d -> train=%d val=%d test=%d", stem, n, len(tr), len(va), len(te))

    man_path = out_root / "split_811_manifest.tsv"
    if not args.dry_run:
        man_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    # 训练用：train 目录下全部 jsonl 路径列表
    list_path = out_root / "train_jsonl_paths.txt"
    if not args.dry_run:
        paths = sorted((out_root / "train").glob("*.jsonl"))
        list_path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

    logger.info(
        "完成: 处理 %d 个已有文件, 缺失 %d 个; 输出 %s",
        done,
        missing,
        out_root,
    )
    if args.dry_run:
        print("\n".join(manifest_rows))


if __name__ == "__main__":
    main()
