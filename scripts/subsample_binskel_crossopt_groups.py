#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按函数身份对齐 O0/O1/O2/O3/Os 五档 jsonl，按比例**成组**抽样写出新 jsonl。

与 collate / cross_opt_pairs 约定一致：函数键为
``(source_file or binary路径, func_name)``——这里优先 ``source_file``，空则退回 ``binary``
（见 ``binskel_jsonl`` 注释：避免binary里带 `_o0` 导致同函数误判）。

示例（数据在 ``/data/chaoni/WQM/datasets``）::

  python scripts/subsample_binskel_crossopt_groups.py \\
    --data-dir /data/chaoni/WQM/datasets \\
    --basename Qwen3.6_C_train \\
    --fraction 0.2 --seed 42 \\
    --out-basename Qwen3.6_C_train_sub20

  python scripts/subsample_binskel_crossopt_groups.py \\
    --data-dir /data/chaoni/WQM/datasets \\
    --basename Qwen3.6_C_val \\
    --fraction 0.2 --seed 42 \\
    --out-basename Qwen3.6_C_val_sub20

会得到 ``Qwen3.6_C_train_sub20_o0.jsonl`` … ``_os.jsonl`` 各一份。

``AB_Q3H2_train_*`` 同理把 ``--basename`` 换成 ``AB_Q3H2_train`` 即可。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

OPT_LEVELS: Tuple[str, ...] = ("o0", "o1", "o2", "o3", "os")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _func_identity(rec: Mapping[str, object]) -> Tuple[str, str]:
    src = str(rec.get("source_file", "") or "").strip()
    binary = str(rec.get("binary", "") or "").strip()
    fname = str(rec.get("func_name", "") or "").strip()
    return (src or binary), fname


def _load_opt_file(
    path: Path,
) -> MutableMapping[Tuple[str, str], Tuple[int, str]]:
    """同一文件内 `(identity,)` → (行号, 原始文本)。重复键后来的覆盖并报 warning。"""
    out: Dict[Tuple[str, str], Tuple[int, str]] = {}
    dup = 0
    bad = 0
    lineno = 0
    raw = path.read_bytes()
    text = raw.decode("utf-8").splitlines()
    for lineno, line in enumerate(text, start=1):
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            bad += 1
            continue
        if not isinstance(rec, dict):
            bad += 1
            continue
        key = _func_identity(rec)
        if key[1] == "":
            logger.warning("%s:%d 缺少 func_name，跳过", path.name, lineno)
            bad += 1
            continue
        if key[0] == "":
            logger.warning("%s:%d 缺少 source_file 与 binary，跳过", path.name, lineno)
            bad += 1
            continue
        if key in out:
            dup += 1
        out[key] = (lineno, line)
    if dup:
        logger.warning("%s: 重复函数键覆盖 %d 次", path.name, dup)
    if bad:
        logger.warning("%s: 跳过坏行或无键 %d", path.name, bad)
    return out


def _intersect_complete_groups(
    by_opt: Mapping[str, Mapping[Tuple[str, str], Tuple[int, str]]],
    levels: Sequence[str],
) -> List[Tuple[str, str]]:
    """仅保留在五档里都出现的键。"""
    if len(levels) < 2:
        raise ValueError("至少需要两档优化级")
    first = levels[0]
    keys = list(by_opt[first].keys())
    cand = set(keys)
    for lvl in levels[1:]:
        cand &= set(by_opt[lvl].keys())
    n_all_first = len(keys)
    n_complete = len(cand)
    if n_complete == 0:
        raise RuntimeError(
            "五档交集为空：请核对是否同一basename、同源数据，以及 source_file/func_name 是否与档无关。"
        )
    logger.info(
        "档位 %s: 首文件键数=%d，五档齐套键数=%d",
        ",".join(levels),
        n_all_first,
        n_complete,
    )
    return sorted(cand)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="含 `{basename}_{o0|…|os}.jsonl` 的目录",
    )
    ap.add_argument(
        "--basename",
        type=str,
        required=True,
        help="文件名前缀（不含 `_o0`），如 Qwen3.6_C_train / Qwen3.6_C_val / AB_Q3H2_train",
    )
    ap.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="仅在「五档齐备」的函数键上按比例抽样，范围 (0,1]",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-basename",
        type=str,
        required=True,
        help="输出前缀，将写出 `{out-basename}_{o0..os}.jsonl` 到 `--data-dir`",
    )
    ap.add_argument(
        "--levels",
        type=str,
        default="o0,o1,o2,o3,os",
        help="逗号分隔档名（须与文件名后缀一致）；默认五档全套",
    )
    args = ap.parse_args()

    f = float(args.fraction)
    if not (0.0 < f <= 1.0):
        logger.error("--fraction 须在 (0,1]，收到 %s", args.fraction)
        sys.exit(1)

    levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())
    if not levels:
        sys.exit("--levels 为空")

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        logger.error("--data-dir 不是目录: %s", data_dir)
        sys.exit(1)

    base = args.basename.strip()
    by_opt: Dict[str, Dict[Tuple[str, str], Tuple[int, str]]] = {}
    for lvl in levels:
        fp = data_dir / f"{base}_{lvl}.jsonl"
        if not fp.is_file():
            logger.error("缺少文件: %s", fp)
            sys.exit(1)
        by_opt[lvl] = dict(_load_opt_file(fp))

    complete = _intersect_complete_groups(by_opt, levels)

    rnd = random.Random(int(args.seed))
    rnd.shuffle(complete)
    if f >= 1.0:
        n_pick = len(complete)
    else:
        n_pick = max(1, math.floor(len(complete) * f))
        n_pick = min(n_pick, len(complete))
    picked = set(complete[:n_pick])

    logger.info(
        "抽样函数组: %d / %d (floor(N×%.4f)==%d)%s",
        len(picked),
        len(complete),
        f,
        n_pick,
        "；已满则取全集" if n_pick >= len(complete) else "",
    )

    out_base = args.out_basename.strip()
    for lvl in levels:
        outp = data_dir / f"{out_base}_{lvl}.jsonl"
        lines_out: List[str] = []
        for key in sorted(picked):
            _, rawline = by_opt[lvl][key]
            lines_out.append(rawline if rawline.endswith("\n") else rawline)
        outp.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        logger.info("写入 %s (%d 行)", outp.name, len(lines_out))


if __name__ == "__main__":
    main()
