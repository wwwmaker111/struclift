#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从同一前缀、五档优化（o0/o1/o2/o3/os）的 jsonl 中按「函数」成组抽样。

默认（``--sample-mode per_project``）::

  - **身份键**（「原函数唯一」）：优先 ``(source_file, func_name)``；无则
    ``(binary, func_name)``。同 jsonl 内同键多行只保留**最后一行**。
  - **配额**：先只看 **该 split 的 o0 文件**（如 ``*_train_o0.jsonl``），统计
    每个「项目」下的**原始行数**（同一键多行会重复计入，对应总样本量）。
    每项目配额 ``round(fraction * 该项目的原始行数)``；各项目配额之和约等于
    ``round(fraction * o0 总行数)``（例如 7.3 万 × 20% ≈ 1.46 万）。
  - **抽样**：仅在 **五档交集** 内的键上，按项目从交集键中无放回随机抽取，
    抽取条数 ``min(该项目的配额, 该项目在交集中的键数)``。
  - 选中键后，在五个优化档 jsonl 各写一行。

可选 ``--sample-mode global``：在全局五档交集上直接抽 ``fraction`` 比例的**键**
（旧逻辑，总量约为「交集键数 × fraction」）。

示例::

  python scripts/subsample_binskel_crossopt_fraction.py \\
    --base-dir /data/chaoni/WQM/datasets \\
    --name-prefix Qwen3.6_C \\
    --fraction 0.2 --seed 42 \\
    --output-suffix _p20

若输入为 ``*_sft.jsonl``（例如 ``CD_train_o0_sft.jsonl``），加 ``--stem-extra _sft``，
输出为 ``CD_train_o0_sft_p20.jsonl``。

输出示例::

  Qwen3.6_C_train_o0_p20.jsonl … Qwen3.6_C_train_os_p20.jsonl
  Qwen3.6_C_val_o0_p20.jsonl   … Qwen3.6_C_val_os_p20.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPT_ORDER: Tuple[str, ...] = ("o0", "o1", "o2", "o3", "os")

FuncKey = Tuple[str, str]


def _source_fn_key(d: dict) -> FuncKey:
    """原函数级键：优先 source_file；否则用 binary（与旧版 binary-only 行兼容）。"""
    sf = str(d.get("source_file", "") or "").strip()
    fn = str(d.get("func_name", "") or "").strip()
    if sf:
        return (sf, fn)
    bn = str(d.get("binary", "") or "").strip()
    return (bn, fn)


def _project_id_from_path(path_str: str, anchor: str, depth: int) -> str:
    """从绝对/相对路径中截取项目 id；找不到 anchor 时归为 __NO_ANCHOR__。"""
    if not path_str.strip():
        return "__EMPTY_PATH__"
    parts = path_str.replace("\\", "/").split("/")
    low = [p.lower() for p in parts]
    anc = anchor.lower()
    try:
        i = low.index(anc)
    except ValueError:
        return "__NO_ANCHOR__"
    segs = parts[i + 1 : i + 1 + max(1, depth)]
    if not segs:
        return "__NO_ANCHOR__"
    return "/".join(segs)


def _project_id_for_key(k: FuncKey, anchor: str, depth: int) -> str:
    return _project_id_from_path(k[0], anchor, depth)


def _load_jsonl_map(
    path: Path,
    *,
    progress_every: int = 0,
    verbose_duplicate_keys: bool = False,
) -> Dict[FuncKey, str]:
    """读入 jsonl，键 → 原始行；同文件内同键多行时默认只保留最后一行。"""
    out: Dict[FuncKey, str] = {}
    dup_count = 0
    dup_sample: List[FuncKey] = []
    n_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f, start=1):
            n_lines = li
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("%s:%d JSON 错误: %s", path, li, e)
                raise
            k = _source_fn_key(d)
            if not k[1]:
                logger.warning("%s:%d 缺少 func_name，跳过", path, li)
                continue
            if k in out:
                dup_count += 1
                if verbose_duplicate_keys:
                    logger.warning("%s: 重复键 %r，保留后者", path.name, k)
                elif len(dup_sample) < 5:
                    dup_sample.append(k)
            out[k] = line
            if progress_every > 0 and li % progress_every == 0:
                logger.info("  %s: 已扫描 %d 行，当前唯一种键数=%d …", path.name, li, len(out))
    if dup_count:
        extra = ""
        if not verbose_duplicate_keys and dup_sample:
            extra = f"，示例键（最多5个）={dup_sample!r}"
        logger.warning(
            "%s: 唯一种键数=%d（扫描行数=%d）；同键重复 %d 次，均保留最后一次%s",
            path.name,
            len(out),
            n_lines,
            dup_count,
            extra,
        )
    else:
        logger.info("%s: 唯一种键数=%d（扫描行数=%d）", path.name, len(out), n_lines)
    return out


def _intersection_keys(maps: List[Dict[FuncKey, str]]) -> set[FuncKey]:
    if not maps:
        return set()
    s = set(maps[0].keys())
    for m in maps[1:]:
        s &= set(m.keys())
    return s


def _stable_rng_seed(base_seed: int, salt: str) -> int:
    """跨进程稳定子种子（不用内置 hash，以免 PYTHONHASHSEED 影响可复现性）。"""
    h = zlib.adler32(salt.encode("utf-8")) & 0xFFFFFFFF
    return (base_seed ^ h ^ (len(salt) << 16)) & 0x7FFFFFFF


def _sample_count(n: int, fraction: float) -> int:
    if n <= 0 or fraction <= 0:
        return 0
    if fraction >= 1.0:
        return n
    k_take = max(1, int(math.floor(float(fraction) * n)))
    return min(k_take, n)


def _count_o0_lines_per_project(
    path: Path,
    anchor: str,
    depth: int,
    progress_every: int,
) -> Tuple[Dict[str, int], int]:
    """按 o0 文件**每一有效行**计数（含同键重复），用于行数配额。

    仅统计能解析且含 func_name 的行；解析失败或缺 func_name 不计入。
    """
    counts: DefaultDict[str, int] = defaultdict(int)
    total = 0
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            k = _source_fn_key(d)
            if not k[1]:
                bad += 1
                continue
            pid = _project_id_for_key(k, anchor, depth)
            counts[pid] += 1
            total += 1
            if progress_every > 0 and li % progress_every == 0:
                logger.info("  %s(o0 配额扫描): 已读 %d 物理行，有效计行行数=%d …", path.name, li, total)
    if bad:
        logger.warning("%s(o0 配额扫描): 跳过 %d 行（JSON 错或缺 func_name）", path.name, bad)
    return dict(counts), total


def _quota_round(fraction: float, raw_n: int) -> int:
    if raw_n <= 0 or fraction <= 0:
        return 0
    if fraction >= 1.0:
        return raw_n
    return int(round(float(fraction) * raw_n))


def _sample_keys_global(keys: Iterable[FuncKey], fraction: float, seed: int) -> List[FuncKey]:
    key_list = sorted(set(keys))
    if not key_list:
        return []
    rng = random.Random(int(seed))
    n = len(key_list)
    k_take = _sample_count(n, fraction)
    rng.shuffle(key_list)
    return sorted(key_list[:k_take])


def _sample_keys_per_project_line_quota(
    keys: Iterable[FuncKey],
    fraction: float,
    seed: int,
    anchor: str,
    depth: int,
    o0_path: Path,
    progress_every: int,
) -> List[FuncKey]:
    """按 o0 文件各项目**原始行数 × fraction** 为配额，在交集中按项目无放回抽样键。"""
    raw_by_proj, raw_total = _count_o0_lines_per_project(
        o0_path, anchor, depth, progress_every
    )
    target_total = _quota_round(fraction, raw_total)
    logger.info(
        "  o0 有效行数总计=%d → round(fraction×行数)=约 %d 条（fraction=%s）",
        raw_total,
        target_total,
        fraction,
    )

    inter = sorted(set(keys))
    by_proj: DefaultDict[str, List[FuncKey]] = defaultdict(list)
    for k in inter:
        pid = _project_id_for_key(k, anchor, depth)
        by_proj[pid].append(k)

    picked: List[FuncKey] = []
    sum_quota = 0
    sum_take = 0
    for proj in sorted(set(raw_by_proj.keys()) | set(by_proj.keys())):
        raw_n = int(raw_by_proj.get(proj, 0))
        quota = _quota_round(fraction, raw_n)
        sum_quota += quota
        lst = sorted(by_proj.get(proj, []))
        n_keys = len(lst)
        take_n = min(quota, n_keys)
        rng_p = random.Random(_stable_rng_seed(int(seed), proj))
        sh = lst[:]
        rng_p.shuffle(sh)
        chosen = sh[:take_n]
        picked.extend(chosen)
        sum_take += len(chosen)
        short = quota - take_n if quota > take_n else 0
        logger.info(
            "  项目 %r: o0 行数=%d 配额=%d | 交集中键数=%d → 实际抽取=%d（配额不足少 %d）",
            proj,
            raw_n,
            quota,
            n_keys,
            take_n,
            short,
        )

    logger.info(
        "  各项目配额之和=%d，实际抽出键数=%d（目标约 %d）",
        sum_quota,
        sum_take,
        target_total,
    )
    return sorted(picked)


def _write_split(
    base_dir: Path,
    name_prefix: str,
    split: str,
    fraction: float,
    seed: int,
    output_suffix: str,
    dry_run: bool,
    progress_every: int = 0,
    verbose_duplicate_keys: bool = False,
    *,
    stem_extra: str = "",
    sample_mode: str = "per_project",
    project_anchor: str = "source_datasets",
    project_depth: int = 1,
) -> None:
    paths = [
        base_dir / f"{name_prefix}_{split}_{tag}{stem_extra}.jsonl"
        for tag in OPT_ORDER
    ]
    for p in paths:
        if not p.is_file():
            logger.error("文件不存在: %s", p)
            raise SystemExit(1)

    maps = [
        _load_jsonl_map(
            p,
            progress_every=progress_every,
            verbose_duplicate_keys=verbose_duplicate_keys,
        )
        for p in paths
    ]
    sizes = [len(m) for m in maps]
    inter = _intersection_keys(maps)
    logger.info(
        "[%s] 各档唯一种键数=%s 五档交集中函数键数=%d (sample_mode=%s)",
        split,
        sizes,
        len(inter),
        sample_mode,
    )
    if len(inter) == 0:
        logger.error(
            "[%s] 五档交集为空：检查 source_file/func_name 是否在五个文件中对齐",
            split,
        )
        raise SystemExit(1)

    if sample_mode == "global":
        picked = _sample_keys_global(inter, fraction, seed)
    elif sample_mode == "per_project":
        o0_path = paths[0]
        logger.info(
            "[%s] 按项目 + o0 行数配额抽样: anchor=%r depth=%d seed=%d o0=%s",
            split,
            project_anchor,
            project_depth,
            seed,
            o0_path.name,
        )
        picked = _sample_keys_per_project_line_quota(
            inter,
            fraction,
            seed,
            project_anchor,
            project_depth,
            o0_path,
            progress_every,
        )
    else:
        logger.error("未知 sample_mode: %s", sample_mode)
        raise SystemExit(1)

    logger.info(
        "[%s] fraction=%s seed=%d → 最终抽样函数键数=%d（交集中共 %d 个键）",
        split,
        fraction,
        seed,
        len(picked),
        len(inter),
    )
    if dry_run:
        logger.info("[%s] dry-run 不写文件", split)
        return

    for tag, p_in, m in zip(OPT_ORDER, paths, maps):
        out = base_dir / f"{name_prefix}_{split}_{tag}{stem_extra}{output_suffix}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as fo:
            for k in picked:
                if k not in m:
                    logger.error("逻辑错误：键 %r 在 %s 缺失", k, p_in.name)
                    raise SystemExit(1)
                fo.write(m[k] + "\n")
        logger.info("已写 %s（%d 行）", out.name, len(picked))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="五档 cross-opt：默认按 o0 行数×fraction 分项目配额，再抽交集键",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="数据集目录（含 *_train_o0.jsonl 等）",
    )
    ap.add_argument(
        "--name-prefix",
        type=str,
        default="Qwen3.6_C",
        help="文件名前缀，例如 Qwen3.6_C → Qwen3.6_C_train_o0.jsonl",
    )
    ap.add_argument("--fraction", type=float, default=0.2, help="抽样比例 (0,1]，默认 0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="_p20",
        help="输出文件名后缀，如 _p20 → Qwen3.6_C_train_o0_p20.jsonl",
    )
    ap.add_argument(
        "--stem-extra",
        type=str,
        default="",
        metavar="SUF",
        help="插在档名与 .jsonl 之间，用于 *_sft.jsonl：传 _sft 即读 CD_train_o0_sft.jsonl，"
        "写出 CD_train_o0_sft_p20.jsonl（默认空=旧行为 CD_train_o0.jsonl）",
    )
    ap.add_argument(
        "--sample-mode",
        choices=("per_project", "global"),
        default="per_project",
        help="per_project=按 o0 各行所属项目计行数×fraction 作配额（默认）；"
        "global=全交集键×fraction（旧逻辑）",
    )
    ap.add_argument(
        "--project-anchor",
        type=str,
        default="source_datasets",
        metavar="SEG",
        help="路径中此目录名之后截项目 id（默认 source_datasets）",
    )
    ap.add_argument(
        "--project-depth",
        type=int,
        default=1,
        metavar="N",
        help="anchor 之后取几段路径作为项目 id（默认 1=常对应顶层 6 个项目名）",
    )
    ap.add_argument("--train-only", action="store_true", help="只处理 train")
    ap.add_argument("--val-only", action="store_true", help="只处理 val")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印统计/抽样结果，不写输出 jsonl",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        metavar="N",
        help="每扫 N 行打一条进度（默认 100000；0=关闭）。",
    )
    ap.add_argument(
        "--verbose-duplicate-keys",
        action="store_true",
        help="同一文件内重复键逐条打 WARNING",
    )
    args = ap.parse_args()

    if not (0.0 < float(args.fraction) <= 1.0):
        logger.error("fraction 须在 (0,1] 内")
        raise SystemExit(1)
    if args.train_only and args.val_only:
        logger.error("不要同时 --train-only 与 --val-only")
        raise SystemExit(1)

    base = args.base_dir.expanduser().resolve()
    suf = args.output_suffix
    stem_extra = str(args.stem_extra or "")
    do_train = not args.val_only
    do_val = not args.train_only

    pe = int(args.progress_every)
    vd = bool(args.verbose_duplicate_keys)
    sm = str(args.sample_mode)
    pa = str(args.project_anchor)
    pd = int(args.project_depth)

    if do_train:
        _write_split(
            base,
            args.name_prefix,
            "train",
            args.fraction,
            args.seed,
            suf,
            args.dry_run,
            progress_every=pe,
            verbose_duplicate_keys=vd,
            stem_extra=stem_extra,
            sample_mode=sm,
            project_anchor=pa,
            project_depth=pd,
        )
    if do_val:
        _write_split(
            base,
            args.name_prefix,
            "val",
            args.fraction,
            args.seed,
            suf,
            args.dry_run,
            progress_every=pe,
            verbose_duplicate_keys=vd,
            stem_extra=stem_extra,
            sample_mode=sm,
            project_anchor=pa,
            project_depth=pd,
        )


if __name__ == "__main__":
    main()
