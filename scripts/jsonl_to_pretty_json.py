#!/usr/bin/env python3
"""将 jsonl 转为缩进 JSON 数组（用于人工查看；大文件慎用内存）。"""
from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: jsonl_to_pretty_json.py <输入.jsonl> <输出.json>", file=sys.stderr)
        sys.exit(1)
    path_in, path_out = sys.argv[1], sys.argv[2]
    objs = []
    with open(path_in, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objs.append(json.loads(line))
    with open(path_out, "w", encoding="utf-8") as fo:
        json.dump(objs, fo, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
