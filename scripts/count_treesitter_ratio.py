#!/usr/bin/env python3
"""
统计数据集中 tree-sitter 解析成功比例

启发式判断：
  - tree-sitter 样本：ast_types 中存在非 15 的值（2=if, 3=while, 8=return 等）
  - 行级回退样本：ast_types 全为 15（unknown），且 src_edge_index 为空

用法:
  python scripts/count_treesitter_ratio.py binskel_zlib_o0.jsonl
  python scripts/count_treesitter_ratio.py binskel_*.jsonl
  python scripts/count_treesitter_ratio.py ~/structlift_datasets/binskel_sqlite_o0.jsonl
  python scripts/count_treesitter_ratio.py file1.jsonl file2.jsonl ...
"""

import json
import sys
from pathlib import Path


def is_treesitter_sample(d: dict) -> bool:
    """判断样本是否来自 tree-sitter 路径（非行级回退）"""
    ast_types = d.get("src_ast_types") or []
    src_edges = d.get("src_edge_index") or [[], []]
    edge_nonempty = bool(src_edges and (src_edges[0] or src_edges[1]))
    if not ast_types:
        return False
    # 行级回退：ast_types 全为 15，且 src_edge_index 为空
    all_unknown = all(t == 15 for t in ast_types)
    if all_unknown and not edge_nonempty:
        return False  # 回退
    return True  # tree-sitter


def main():
    paths = sys.argv[1] if len(sys.argv) > 1 else None
    if not paths:
        print("用法: python scripts/count_treesitter_ratio.py <file.jsonl> [file2.jsonl ...]")
        print("  或: python scripts/count_treesitter_ratio.py binskel_*.jsonl")
        sys.exit(1)

    from glob import glob
    files = []
    for p in sys.argv[1:]:
        files.extend(glob(p) if "*" in p else [p])
    files = [f for f in files if Path(f).exists()]

    if not files:
        print("未找到文件")
        sys.exit(1)

    total = 0
    n_ts = 0
    n_fb = 0
    by_file = {}

    for fp in files:
        n_ts_f = 0
        n_fb_f = 0
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if is_treesitter_sample(d):
                    n_ts += 1
                    n_ts_f += 1
                else:
                    n_fb += 1
                    n_fb_f += 1
        by_file[Path(fp).name] = (n_ts_f, n_fb_f)

    print("=" * 60)
    print("tree-sitter 解析成功比例统计")
    print("=" * 60)
    print(f"\n总样本数: {total}")
    print(f"  tree-sitter (语句级): {n_ts}")
    print(f"  行级回退:            {n_fb}")
    if total > 0:
        pct = 100 * n_ts / total
        print(f"\ntree-sitter 成功率: {pct:.1f}%")
    print("\n按文件:")
    for name, (ts, fb) in sorted(by_file.items()):
        s = ts + fb
        pct = 100 * ts / s if s else 0
        print(f"  {name}: {ts} tree-sitter, {fb} 回退 ({pct:.1f}% tree-sitter)")
    print()


if __name__ == "__main__":
    main()
