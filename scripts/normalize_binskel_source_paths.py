#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式改写 binskel jsonl 中的 ``source_file``（及可选 ``binary``）路径前缀。

用于「导出机在 /mnt/e/structlift_src_datasets，服务器在 /data/.../source_datasets」
这一类跨机器绝对路径漂移，便于在 node4 上再跑 ``augment_binskel_sft`` 或开源码审计。

不解析 tokenizer；仅替换字符串前缀（逐条）。

示例::

  python3 scripts/normalize_binskel_source_paths.py \\
    --input /data/chaoni/WQM/datasets/Qwen3.6_C_train_o0.jsonl \\
    --output /data/chaoni/WQM/datasets/Qwen3.6_C_train_o0_norm.jsonl \\
    --rewrite-prefix /mnt/e/structlift_src_datasets /data/chaoni/WQM/source_datasets

可多次传入 ``--rewrite-prefix FROM TO``，按给定顺序套用（仅当当前字符串以 FROM 开头时替换）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _rewrite_one(s: str, rewrites: List[Tuple[str, str]]) -> str:
    if not isinstance(s, str) or not s:
        return s
    x = s.replace("\\", "/")
    for frm, to in rewrites:
        ff = str(frm).replace("\\", "/").rstrip("/")
        tt = str(to).replace("\\", "/").rstrip("/")
        if ff and x.startswith(ff):
            x = tt + x[len(ff) :]
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="批量改写 jsonl 内 source_file/binary 前缀")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--rewrite-prefix",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        dest="pairs",
        help="前缀替换 FROM→TO（可重复；顺序生效）",
    )
    ap.add_argument("--also-binary", action="store_true", help="同时对 binary 字段做相同替换")
    ap.add_argument("--dry-run", action="store_true", help="不写文件；只日志统计替换条数")
    args = ap.parse_args()

    pairs = list(args.pairs or [])
    if not pairs:
        logger.error("请至少传一组 --rewrite-prefix FROM TO")
        raise SystemExit(1)

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    if not inp.is_file():
        logger.error("输入不存在: %s", inp)
        raise SystemExit(1)
    if not args.dry_run:
        outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_sf_chg = 0
    n_bn_chg = 0

    fout = outp.open("w", encoding="utf-8") if not args.dry_run else None
    try:
        with inp.open("r", encoding="utf-8") as fin:
            for li, line in enumerate(fin, start=1):
                raw = line.rstrip("\n\r")
                if not raw.strip():
                    continue
                d = json.loads(raw)
                n += 1
                sf0 = str(d.get("source_file") or "")
                sf1 = _rewrite_one(sf0, pairs)
                if sf1 != sf0:
                    d["source_file"] = sf1
                    n_sf_chg += 1
                if args.also_binary and d.get("binary") is not None:
                    b0 = str(d.get("binary") or "")
                    b1 = _rewrite_one(b0, pairs)
                    if b1 != b0:
                        d["binary"] = b1
                        n_bn_chg += 1
                if fout:
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                if li % 20000 == 0:
                    logger.info("  已处理 %d 行 …", li)
    finally:
        if fout:
            fout.close()

    logger.info(
        "完成: 记录=%d, source_file 改写=%d, binary 改写=%d → %s",
        n,
        n_sf_chg,
        n_bn_chg,
        "dry-run" if args.dry_run else str(outp),
    )


if __name__ == "__main__":
    main()
