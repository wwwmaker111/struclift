#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 9 项目 × 5 优化档（45 个）binskel jsonl 按 8:1:1 划分，合并为三个 jsonl。

- **不修改、不覆盖** 源目录中的任何文件；只读取并写入新目录。
- 对每个源文件内的样本行独立打乱（可复现 seed），再按 8:1:1 切分，
  最后把所有文件的 train 段合并为 train.jsonl，val / test 同理。

用法（在仓库根或任意目录）::

  python scripts/split_9proj_5opt_to_model_jsonl.py \\
    --src-dir E:/structlift_dataset \\
    --out-dir E:/structlift_model_dataset \\
    --seed 42

若某预期文件名不存在，打印警告并跳过（可用 --strict 要求 45 个必须齐）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 与 docs/build_binskel_9x5_full.md 一致
SINGLE_ELF = ("zlib", "curl", "sqlite", "busybox", "openssl", "libxml2")
MULTIBIN = ("openssh", "ffmpeg", "coreutils")


def expected_45_filenames() -> List[str]:
    out: List[str] = []
    for p in SINGLE_ELF:
        for o in ("o0", "o1", "o2", "o3"):
            out.append(f"binskel_{p}_{o}.jsonl")
        out.append(f"binskel_{p}_os.jsonl")
    for p in MULTIBIN:
        for o in ("o0", "o1", "o2", "o3"):
            out.append(f"binskel_{p}_{o}_multibin.jsonl")
        out.append(f"binskel_{p}_os_multibin.jsonl")
    assert len(out) == 45, len(out)
    return out


def split_counts(n: int) -> Tuple[int, int, int]:
    """8:1:1，保证三段之和为 n。"""
    if n <= 0:
        return 0, 0, 0
    n_train = (n * 8) // 10
    n_val = (n * 1) // 10
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def find_file(src: Path, name: str) -> Path | None:
    """先在根目录找，再在子目录浅层找。"""
    direct = src / name
    if direct.is_file():
        return direct
    for p in src.glob("**/" + name):
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="45 个 jsonl → train/val/test 三个 jsonl（8:1:1）")
    ap.add_argument("--src-dir", type=Path, required=True, help="含 45 个 binskel jsonl 的目录（只读）")
    ap.add_argument("--out-dir", type=Path, required=True, help="输出目录（将创建 train.jsonl / val.jsonl / test.jsonl）")
    ap.add_argument("--seed", type=int, default=42, help="每文件内打乱用种子（可复现）")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="要求 45 个预期文件全部存在，否则退出码 1",
    )
    ap.add_argument(
        "--add-source-tag",
        action="store_true",
        help="在每条 JSON 中写入 _struclift_source_jsonl 便于溯源（默认不写）",
    )
    args = ap.parse_args()

    src_dir = args.src_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not src_dir.is_dir():
        print(f"错误: 源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"

    names = expected_45_filenames()
    rng_master = random.Random(args.seed)

    train_lines: List[str] = []
    val_lines: List[str] = []
    test_lines: List[str] = []

    missing: List[str] = []
    used_paths: List[Path] = []

    for name in names:
        path = find_file(src_dir, name)
        if path is None:
            missing.append(name)
            continue
        used_paths.append(path)

        raw_lines: List[str] = []
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_lines.append(line)

        n = len(raw_lines)
        if n == 0:
            print(f"  跳过(空): {path}", flush=True)
            continue

        # 每文件独立 RNG，避免与文件顺序耦合；种子与全局 seed、文件名绑定
        file_seed = args.seed + hash(name) % (2**31)
        rng = random.Random(file_seed)
        rng.shuffle(raw_lines)

        nt, nv, ns = split_counts(n)
        assert nt + nv + ns == n

        chunk_train = raw_lines[:nt]
        chunk_val = raw_lines[nt : nt + nv]
        chunk_test = raw_lines[nt + nv :]

        if args.add_source_tag:

            def tag_line(s: str) -> str:
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    return s
                obj["_struclift_source_jsonl"] = name
                return json.dumps(obj, ensure_ascii=False)

            chunk_train = [tag_line(s) for s in chunk_train]
            chunk_val = [tag_line(s) for s in chunk_val]
            chunk_test = [tag_line(s) for s in chunk_test]

        train_lines.extend(chunk_train)
        val_lines.extend(chunk_val)
        test_lines.extend(chunk_test)

        print(f"  {name}: n={n} → train={nt} val={nv} test={ns}", flush=True)

    if missing:
        print(f"\n警告: 未找到 {len(missing)} 个预期文件（已跳过）:", flush=True)
        for m in missing[:20]:
            print(f"    {m}", flush=True)
        if len(missing) > 20:
            print(f"    ... 共 {len(missing)} 个", flush=True)
        if args.strict:
            print("错误: --strict 要求 45 个文件齐全。", file=sys.stderr)
            sys.exit(1)

    # 合并后整体再打乱一次，避免按项目块顺序（可选）
    rng_master.shuffle(train_lines)
    rng_master.shuffle(val_lines)
    rng_master.shuffle(test_lines)

    def write_jsonl(path: Path, lines: List[str]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")

    write_jsonl(train_path, train_lines)
    write_jsonl(val_path, val_lines)
    write_jsonl(test_path, test_lines)

    print(f"\n完成。源目录未修改: {src_dir}")
    print(f"已写入:")
    print(f"  {train_path}  ({len(train_lines)} 行)")
    print(f"  {val_path}   ({len(val_lines)} 行)")
    print(f"  {test_path}  ({len(test_lines)} 行)")
    print(f"已使用源文件数: {len(used_paths)} / 45")


if __name__ == "__main__":
    main()
