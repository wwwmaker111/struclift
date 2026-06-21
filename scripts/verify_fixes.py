#!/usr/bin/env python3
"""验证 bin_edge_type 5种边类型 + src_edge_index 非空"""
import json
from collections import Counter

datasets = {
    'zlib': '/home/wuqiongmin/structlift_datasets/binskel_zlib_o0.jsonl',
    'sqlite': '/home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl',
    'busybox': '/home/wuqiongmin/structlift_datasets/binskel_busybox_o0.jsonl',
    'openssl': '/home/wuqiongmin/structlift_datasets/binskel_openssl_o0.jsonl',
}

EDGE_NAMES = {0: 'COND', 1: 'UNCOND', 2: 'BACK', 3: 'CALL', 4: 'RET'}

for name, path in datasets.items():
    total = 0
    edge_counter = Counter()
    src_edge_nonempty = 0
    src_edge_total = 0
    bin_edge_nonempty = 0

    with open(path) as f:
        for line in f:
            s = json.loads(line)
            total += 1
            for et in s.get('bin_edge_type', []):
                edge_counter[et] += 1
            bei = s.get('bin_edge_index', [])
            if bei and len(bei) == 2 and len(bei[0]) > 0:
                bin_edge_nonempty += 1
            sei = s.get('src_edge_index', [])
            if sei and len(sei) == 2 and len(sei[0]) > 0:
                src_edge_nonempty += 1
                src_edge_total += len(sei[0])

    print(f"=== {name} ({total} samples) ===")
    print(f"  bin_edge_index 非空: {bin_edge_nonempty}/{total}")
    print(f"  bin_edge_type 分布:")
    total_edges = sum(edge_counter.values())
    for eid in sorted(edge_counter.keys()):
        cnt = edge_counter[eid]
        pct = 100 * cnt / total_edges if total_edges else 0
        print(f"    {EDGE_NAMES.get(eid, f'UNK({eid})')}: {cnt} ({pct:.1f}%)")
    print(f"  src_edge_index 非空: {src_edge_nonempty}/{total} ({100*src_edge_nonempty/total:.1f}%)")
    print(f"  src_edge 总数: {src_edge_total}")
    print()
