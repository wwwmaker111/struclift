#!/usr/bin/env python3
"""
合并 E:\\ABCD_datasets\\AB / CD 下的 binskel_*.jsonl。

输出（在 --root 下）：
  - AB_train.jsonl / CD_train.jsonl：排除 openssh、busybox、sqlite 后的 6 项目合并。
  - AB_train_{o0,o1,o2,o3,os}.jsonl / CD_train_*.jsonl：同上 6 项目按优化等级各一份。

不删除、不修改源文件。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OPTS = ("o0", "o1", "o2", "o3", "os")
EXCLUDE = frozenset({"openssh", "busybox", "sqlite"})
# 6 项目训练集（与 EXCLUDE 互补）
TRAIN_PROJECTS = frozenset(
    {"coreutils", "curl", "ffmpeg", "libxml2", "openssl", "zlib"}
)

# binskel_AB_coreutils_o0.jsonl
NAME_RE = re.compile(r"^binskel_(AB|CD)_(.+)_((?:o[0-3])|os)\.jsonl$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("E:/ABCD_datasets"),
        help="含 AB/ 与 CD/ 子目录的根路径",
    )
    return p.parse_args()


def collect_files(split_dir: Path, split: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(split_dir.glob(f"binskel_{split}_*.jsonl")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        proj = m.group(2)
        if proj not in TRAIN_PROJECTS:
            continue
        out.append(p)
    return out


def write_concat(paths: list[Path], out: Path) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fw:
        for f in paths:
            with f.open("r", encoding="utf-8") as fr:
                for line in fr:
                    fw.write(line)
                    n += 1
    return n


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    for split in ("AB", "CD"):
        d = root / split
        if not d.is_dir():
            print(f"[skip] 无目录: {d}")
            continue

        all_train = collect_files(d, split)
        # 全量 6 项目合并
        out_all = root / f"{split}_train.jsonl"
        n_all = write_concat(all_train, out_all)
        print(f"{split}_train.jsonl: files={len(all_train)} lines={n_all} -> {out_all}")

        # 按优化等级
        for opt in OPTS:
            subset = []
            for p in all_train:
                m = NAME_RE.match(p.name)
                if m and m.group(3) == opt:
                    subset.append(p)
            out_opt = root / f"{split}_train_{opt}.jsonl"
            n_opt = write_concat(subset, out_opt)
            print(
                f"{split}_train_{opt}.jsonl: files={len(subset)} lines={n_opt} -> {out_opt}"
            )


if __name__ == "__main__":
    main()
