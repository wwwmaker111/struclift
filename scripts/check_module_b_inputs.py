#!/usr/bin/env python3
"""检查 Module B 所需输入字段在各数据集中的存在和有效性"""
import json

datasets = {
    'zlib': '/home/wuqiongmin/structlift_datasets/binskel_zlib_o0.jsonl',
    'sqlite': '/home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl',
    'busybox': '/home/wuqiongmin/structlift_datasets/binskel_busybox_o0.jsonl',
    'openssl': '/home/wuqiongmin/structlift_datasets/binskel_openssl_o0.jsonl',
}

for name, path in datasets.items():
    n = 0
    has_bin_edge = 0; has_src_edge = 0; has_pattern = 0; has_ast = 0
    has_align_mat = 0; has_align_conf = 0; has_signal = 0
    has_src_tokens = 0; has_struct_feats = 0
    src_edge_nonempty = 0
    bin_edge_nonempty = 0
    total = 0

    with open(path) as f:
        for line in f:
            s = json.loads(line)
            total += 1

            bei = s.get('bin_edge_index', [])
            if bei and len(bei) == 2 and len(bei[0]) > 0:
                bin_edge_nonempty += 1
            sei = s.get('src_edge_index', [])
            if sei and len(sei) == 2 and len(sei[0]) > 0:
                src_edge_nonempty += 1
            if s.get('pattern_labels'):
                has_pattern += 1
            if s.get('src_ast_types'):
                has_ast += 1
            if s.get('alignment_matrix') and any(any(r) for r in s['alignment_matrix']):
                has_align_mat += 1
            if s.get('alignment_confidence') and any(any(r) for r in s['alignment_confidence']):
                has_align_conf += 1
            if s.get('signal_stats'):
                has_signal += 1
            if s.get('src_token_ids') and len(s['src_token_ids']) > 0:
                has_src_tokens += 1
            if s.get('bin_struct_features') and len(s['bin_struct_features']) > 0:
                has_struct_feats += 1

    print(f"=== {name} ({total} samples) ===")
    print(f"  --- 二进制侧 (Module A → B) ---")
    print(f"  bin_edge_index 非空:   {bin_edge_nonempty}/{total} ({100*bin_edge_nonempty/total:.1f}%)  → E_b")
    print(f"  bin_struct_features:   {has_struct_feats}/{total} ({100*has_struct_feats/total:.1f}%)  → H̃_b 的输入")
    print(f"  pattern_labels:        {has_pattern}/{total} ({100*has_pattern/total:.1f}%)  → P_pat")
    print(f"  --- 源码侧 (Module A_src → B) ---")
    print(f"  src_token_ids 非空:    {has_src_tokens}/{total} ({100*has_src_tokens/total:.1f}%)  → H̃_s 的输入")
    print(f"  src_ast_types:         {has_ast}/{total} ({100*has_ast/total:.1f}%)  → T_s")
    print(f"  src_edge_index 非空:   {src_edge_nonempty}/{total} ({100*src_edge_nonempty/total:.1f}%)  → E_s")
    print(f"  --- 对齐监督 (Module B 输出目标) ---")
    print(f"  alignment_matrix 有效: {has_align_mat}/{total} ({100*has_align_mat/total:.1f}%)  → M_gt")
    print(f"  alignment_confidence:  {has_align_conf}/{total} ({100*has_align_conf/total:.1f}%)  → W")
    print(f"  signal_stats:          {has_signal}/{total} ({100*has_signal/total:.1f}%)")
    print()
