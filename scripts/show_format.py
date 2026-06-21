#!/usr/bin/env python3
"""展示数据集格式"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/wuqiongmin/structlift_datasets/binskel_openssl_o0.jsonl"
with open(path) as f:
    s = json.loads(f.readline())

print("=== 顶层字段 ===")
for k, v in s.items():
    t = type(v).__name__
    if isinstance(v, list):
        n = len(v)
        if v and isinstance(v[0], list):
            t = f"list[list] [{n} x {len(v[0])}]"
        elif v and isinstance(v[0], dict):
            t = f"list[dict] [{n}], keys={list(v[0].keys())}"
        elif v and isinstance(v[0], (int, float)):
            t = f"list[{type(v[0]).__name__}] [{n}]"
            if n <= 5:
                t += f" = {v}"
        elif v and isinstance(v[0], str):
            t = f"list[str] [{n}]"
        else:
            t = f"list [{n}]"
    elif isinstance(v, dict):
        t = f"dict, keys={list(v.keys())[:10]}"
    elif isinstance(v, str):
        t = f"str = {repr(v[:80])}"
    elif isinstance(v, (int, float)):
        t = f"{type(v).__name__} = {v}"
    print(f"  {k}: {t}")

print(f"\n=== 维度 ===")
bbs = s.get("basic_blocks", [])
print(f"  BB 数: {len(bbs)}")
am = s.get("alignment_matrix", [])
if am:
    print(f"  alignment_matrix: [{len(am)} x {len(am[0]) if am[0] else 0}] (BB x 语句)")
ac = s.get("alignment_confidence", [])
if ac:
    print(f"  alignment_confidence: [{len(ac)} x {len(ac[0]) if ac[0] else 0}]")
ss = s.get("source_statements_used", [])
if ss:
    print(f"  source_statements_used: [{len(ss)}] 条语句")
    print(f"    每条语句的 keys: {list(ss[0].keys())}")
sig = s.get("signal_stats")
if sig:
    print(f"  signal_stats: {sig}")

print(f"\n=== 第一个 BB 示例 ===")
if bbs:
    bb = bbs[0]
    for k2, v2 in bb.items():
        if isinstance(v2, list) and len(v2) > 8:
            print(f"  {k2}: list[{len(v2)}] (前3: {v2[:3]}...)")
        else:
            print(f"  {k2}: {v2}")

print(f"\n=== 第一条语句示例 ===")
if ss:
    print(f"  {ss[0]}")
