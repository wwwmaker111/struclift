#!/usr/bin/env python3
"""检查 Module C (结构感知代码生成) 训练所需数据字段"""
import json

datasets = {
    'zlib': '/home/wuqiongmin/structlift_datasets/binskel_zlib_o0.jsonl',
    'sqlite': '/home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl',
    'busybox': '/home/wuqiongmin/structlift_datasets/binskel_busybox_o0.jsonl',
    'openssl': '/home/wuqiongmin/structlift_datasets/binskel_openssl_o0.jsonl',
}

for name, path in datasets.items():
    total = 0
    has_src_stmts = 0; has_src_code = 0; has_src_tokens = 0
    has_bin_opcodes = 0; has_bin_bb_texts = 0; has_bin_insn = 0
    has_align = 0; has_conf = 0
    has_pattern = 0; has_struct_feats = 0
    has_bin_edge = 0; has_src_edge = 0
    has_src_ast = 0
    has_func_name = 0; has_src_file = 0

    sample_keys = set()

    with open(path) as f:
        for i, line in enumerate(f):
            s = json.loads(line)
            total += 1
            if i == 0:
                sample_keys = set(s.keys())

            if s.get('src_stmts') and len(s['src_stmts']) > 0:
                has_src_stmts += 1
            if s.get('src_code') and len(s.get('src_code', '')) > 0:
                has_src_code += 1
            if s.get('src_token_ids') and len(s['src_token_ids']) > 0:
                has_src_tokens += 1
            if s.get('bin_opcodes') and len(s['bin_opcodes']) > 0:
                has_bin_opcodes += 1
            if s.get('bin_bb_texts') and len(s['bin_bb_texts']) > 0:
                has_bin_bb_texts += 1
            if s.get('bin_insn_texts') and len(s['bin_insn_texts']) > 0:
                has_bin_insn += 1
            am = s.get('alignment_matrix')
            if am and any(any(r) for r in am):
                has_align += 1
            ac = s.get('alignment_confidence')
            if ac and any(any(r) for r in ac):
                has_conf += 1
            if s.get('pattern_labels') and len(s['pattern_labels']) > 0:
                has_pattern += 1
            if s.get('bin_struct_features') and len(s['bin_struct_features']) > 0:
                has_struct_feats += 1
            bei = s.get('bin_edge_index', [])
            if bei and len(bei) == 2 and len(bei[0]) > 0:
                has_bin_edge += 1
            sei = s.get('src_edge_index', [])
            if sei and len(sei) == 2 and len(sei[0]) > 0:
                has_src_edge += 1
            if s.get('src_ast_types') and len(s['src_ast_types']) > 0:
                has_src_ast += 1
            if s.get('func_name'):
                has_func_name += 1
            if s.get('src_file'):
                has_src_file += 1

    pct = lambda x: f"{x}/{total} ({100*x/total:.1f}%)"

    print(f"=== {name} ({total} samples) ===")
    print(f"  所有字段: {sorted(sample_keys)}")
    print()
    print(f"  ── Module C 训练所需 ──")
    print(f"  [Skeleton 构建] src_stmts:          {pct(has_src_stmts)}  ← 源码语句(构建骨架+GT)")
    print(f"  [Skeleton 构建] src_code:            {pct(has_src_code)}  ← 完整源码")
    print(f"  [Skeleton 构建] src_ast_types:       {pct(has_src_ast)}  ← AST类型(识别控制结构)")
    print(f"  [Binary ctx]    bin_opcodes:         {pct(has_bin_opcodes)}  ← BB指令序列")
    print(f"  [Binary ctx]    bin_bb_texts:        {pct(has_bin_bb_texts)}  ← BB反汇编文本")
    print(f"  [Binary ctx]    bin_insn_texts:      {pct(has_bin_insn)}  ← 指令文本")
    print(f"  [Slot mask]     alignment_matrix:    {pct(has_align)}  ← M(BB↔语句映射)")
    print(f"  [Slot mask]     alignment_confidence:{pct(has_conf)}  ← W(置信度)")
    print(f"  [Graph KV]      bin_struct_features: {pct(has_struct_feats)}  ← 结构特征(→Ĥ_b)")
    print(f"  [Graph KV]      bin_edge_index:      {pct(has_bin_edge)}  ← E_b(二进制CFG)")
    print(f"  [Graph KV]      pattern_labels:      {pct(has_pattern)}  ← P_pat")
    print(f"  [Graph KV]      src_edge_index:      {pct(has_src_edge)}  ← E_s(源码CFG)")
    print(f"  [元数据]        func_name:           {pct(has_func_name)}")
    print(f"  [元数据]        src_file:            {pct(has_src_file)}")
    print()
