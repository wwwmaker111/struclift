#!/usr/bin/env python3
import json

datasets = {
    'zlib': '/home/wuqiongmin/structlift_datasets/binskel_zlib_o0.jsonl',
    'sqlite': '/home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl',
    'busybox': '/home/wuqiongmin/structlift_datasets/binskel_busybox_o0.jsonl',
    'openssl': '/home/wuqiongmin/structlift_datasets/binskel_openssl_o0.jsonl',
}

# EdgeType: 0=COND, 1=UNCOND, 2=BACK, 3=CALL, 4=RET
etype_names = {0: 'cond', 1: 'uncond', 2: 'back', 3: 'call', 4: 'ret'}

for name, path in datasets.items():
    all_etypes = set()
    n_samples = 0
    example = None
    with open(path) as f:
        for i, line in enumerate(f):
            s = json.loads(line)
            n_samples += 1
            for et in s.get('bin_edge_type', []):
                all_etypes.add(et)
            if example is None and len(s.get('bin_block_lengths', [])) >= 5:
                example = s
            if i >= 2000:
                break

    print(f"=== {name} (前{n_samples}条) ===")
    print(f"  edge_types 出现过: {sorted(all_etypes)} -> {[etype_names.get(t, f'?{t}') for t in sorted(all_etypes)]}")

    if example:
        nb = len(example['bin_block_lengths'])
        ne = len(example['bin_edge_type'])
        n_instr = len(example['bin_opcodes'])
        n_ops = len(example['bin_operand_types'])
        sf = example.get('bin_struct_features', [])
        print(f"  示例函数: {example['func_name']}")
        print(f"    V_b (BB节点): {nb} 个, 每个含指令序列 (总指令{n_instr}, 总操作数{n_ops})")
        print(f"    E_b (CFG边):  {ne} 条, edge_types={sorted(set(example['bin_edge_type']))}")
        print(f"    bin_block_lengths: {example['bin_block_lengths'][:8]}{'...' if nb>8 else ''}")
        print(f"    bin_struct_features: [{len(sf)} x {len(sf[0]) if sf else 0}]")
        print(f"    pattern_labels: {example['pattern_labels'][:8]}{'...' if nb>8 else ''}")
    print()
