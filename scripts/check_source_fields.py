#!/usr/bin/env python3
"""检查 source_lines / block_src_lines / source_file 等字段的实际内容"""
import json

datasets = {
    'zlib': '/home/wuqiongmin/structlift_datasets/binskel_zlib_o0.jsonl',
    'sqlite': '/home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl',
    'openssl': '/home/wuqiongmin/structlift_datasets/binskel_openssl_o0.jsonl',
}

for name, path in datasets.items():
    print(f"=== {name} ===")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            s = json.loads(line)
            fn = s.get('func_name', '?')
            print(f"\n  样本 {i}: {fn}")

            sl = s.get('source_lines')
            if sl:
                if isinstance(sl, list):
                    print(f"  source_lines: list[{len(sl)}], 前3项: {sl[:3]}")
                elif isinstance(sl, dict):
                    keys = list(sl.keys())[:3]
                    print(f"  source_lines: dict[{len(sl)}], 前3键: {keys}")
                    for k in keys[:2]:
                        v = sl[k]
                        print(f"    [{k}] = {repr(v)[:120]}")
                else:
                    print(f"  source_lines: {type(sl).__name__} = {repr(sl)[:120]}")

            sf = s.get('source_file')
            if sf:
                print(f"  source_file: {repr(sf)[:200]}")
            else:
                print(f"  source_file: (空)")

            bsl = s.get('block_src_lines')
            if bsl:
                if isinstance(bsl, list):
                    print(f"  block_src_lines: list[{len(bsl)}], 前2项:")
                    for j, item in enumerate(bsl[:2]):
                        print(f"    [{j}] = {repr(item)[:150]}")
                elif isinstance(bsl, dict):
                    keys = list(bsl.keys())[:3]
                    print(f"  block_src_lines: dict[{len(bsl)}], 前2键:")
                    for k in keys[:2]:
                        print(f"    [{k}] = {repr(bsl[k])[:150]}")

            bsc = s.get('block_src_confidence')
            if bsc:
                if isinstance(bsc, list):
                    print(f"  block_src_confidence: list[{len(bsc)}], 前5: {bsc[:5]}")
                elif isinstance(bsc, dict):
                    keys = list(bsc.keys())[:5]
                    print(f"  block_src_confidence: dict[{len(bsc)}], 前5: {[(k, bsc[k]) for k in keys]}")

            src_stl = s.get('src_stmt_lengths')
            if src_stl:
                print(f"  src_stmt_lengths: list[{len(src_stl)}], 前10: {src_stl[:10]}")
            src_tok = s.get('src_token_ids')
            if src_tok:
                print(f"  src_token_ids: list[{len(src_tok)}], 前10: {src_tok[:10]}")
            src_ast = s.get('src_ast_types')
            if src_ast:
                print(f"  src_ast_types: list[{len(src_ast)}], 前10: {src_ast[:10]}")

    print()
