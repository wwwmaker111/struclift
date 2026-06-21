#!/usr/bin/env python3
"""快速检查三项目 pattern_labels 与 bin_edge_type"""
import json
from pathlib import Path

paths = {
    'sqlite': Path.home() / 'structlift_datasets/binskel_sqlite_o0.jsonl',
    'busybox': Path.home() / 'structlift_datasets/binskel_busybox_o0.jsonl',
    'openssl': Path.home() / 'structlift_datasets/binskel_openssl_o0.jsonl',
}

for name, p in paths.items():
    p = p.expanduser()
    if not p.exists():
        print(f"{name}: 文件不存在")
        continue
    n = has_if = has_et = ex = 0
    with open(p) as f:
        for line in f:
            s = json.loads(line)
            n += 1
            pl = s.get('pattern_labels') or []
            et = s.get('bin_edge_type') or []
            if 1 in pl:
                has_if += 1
            if et:
                has_et += 1
            if ex is None and 1 in pl and len(pl) >= 4:
                ex = s
    print(f"\n=== {name} ({n} 样本) ===")
    print(f"  有 IF_HEADER: {has_if}/{n}")
    print(f"  有 bin_edge_type 非空: {has_et}/{n}")
    if ex:
        print(f"  示例: {ex.get('func_name')} pl={ex.get('pattern_labels')[:12]}...")
