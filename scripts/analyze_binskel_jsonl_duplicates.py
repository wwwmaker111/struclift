#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全面分析单份 binskel 风格 jsonl 的「重复」来源。

与 subsample 一致，样本键为 ``(source_file or binary, func_name)``。

输出概要：
  - 行数、有效 JSON、缺字段行数
  - 唯一键数、「同键多行」造成的冗余行数
  - 键出现次数的直方图（1 次 / 2 次 / … / 10+ 次）
  - **整行文本完全相同**的重复（与「同键不同内容」区分）
  - 对「出现次数最多」的一批键，取前两行 JSON 做顶层字段差异对照

用法::

  python3 scripts/analyze_binskel_jsonl_duplicates.py \\
    --jsonl /data/chaoni/WQM/datasets/Qwen3.6_C_train_o0.jsonl

  python3 scripts/analyze_binskel_jsonl_duplicates.py \\
    --jsonl Qwen3.6_C_train_o0.jsonl --top-compare 20 --out report_o0.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FuncKey = Tuple[str, str]


def _func_key(d: dict) -> FuncKey:
    src = str(d.get("source_file", "") or d.get("binary", "") or "")
    fn = str(d.get("func_name", "") or "")
    return (src, fn)


def _stable_json(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _top_level_diff(a: dict, b: dict) -> Tuple[List[str], List[str], List[str]]:
    """返回 (仅 a 有, 仅 b 有, 同键但序列化不同)."""
    ka, kb = set(a.keys()), set(b.keys())
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    common = ka & kb
    diff_val: List[str] = []
    for k in sorted(common):
        if _stable_json(a[k]) != _stable_json(b[k]):
            diff_val.append(k)
    return only_a, only_b, diff_val


def _hist_bucket(c: int) -> str:
    if c <= 0:
        return "0"
    if c >= 10:
        return "10+"
    return str(c)


def main() -> None:
    ap = argparse.ArgumentParser(description="分析 binskel jsonl 重复结构")
    ap.add_argument("--jsonl", type=Path, required=True, help="单份 jsonl 路径")
    ap.add_argument(
        "--top-compare",
        type=int,
        default=25,
        metavar="N",
        help="对「出现次数排名前列」的 N 个键做两两字段对比（默认 25）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="可写到该文件；不传则只打 stdout",
    )
    args = ap.parse_args()

    path: Path = args.jsonl.expanduser().resolve()
    if not path.is_file():
        logger.error("文件不存在: %s", path)
        raise SystemExit(1)

    lines_total = 0
    lines_nonempty = 0
    json_ok = 0
    json_err = 0
    missing_fn = 0
    missing_src = 0
    key_counts: Counter[FuncKey] = Counter()
    # 仅统计「整行规范化后」的重复（去尾部换行差异）
    raw_line_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f, start=1):
            lines_total += 1
            s = line.rstrip("\n\r")
            if not s.strip():
                continue
            lines_nonempty += 1
            raw_line_counts[s] += 1
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                json_err += 1
                logger.warning("%s:%d JSON 解析失败", path.name, li)
                continue
            json_ok += 1
            fn = str(d.get("func_name", "") or "")
            src = str(d.get("source_file", "") or d.get("binary", "") or "")
            if not fn:
                missing_fn += 1
            if not src:
                missing_src += 1
            key_counts[_func_key(d)] += 1

    n_keys = len(key_counts)
    sum_lines = sum(key_counts.values())
    redundant_by_key = sum(c - 1 for c in key_counts.values() if c > 1)

    # 键重数直方图：有多少个「键」出现了 1 次、2 次、…
    mult_hist: Counter[str] = Counter()
    for c in key_counts.values():
        mult_hist[_hist_bucket(c)] += 1

    # 完全雷同行
    exact_line_redundant = sum(cnt - 1 for cnt in raw_line_counts.values() if cnt > 1)
    exact_line_unique_strings = sum(1 for cnt in raw_line_counts.values() if cnt == 1)
    exact_line_dup_strings = sum(1 for cnt in raw_line_counts.values() if cnt > 1)

    lines: List[str] = []
    W = lines.append

    W("=" * 72)
    W(f"文件: {path}")
    W(f"扫描物理行数: {lines_total}  非空行数: {lines_nonempty}")
    W(f"JSON 解析成功: {json_ok}  失败: {json_err}")
    W(f"缺 func_name 的记录行数: {missing_fn}  缺 binary/source_file 的记录行数: {missing_src}")
    W("")
    W("— 按 (source_file|binary, func_name) 统计 —")
    W(f"逻辑样本行数(有效 JSON): {json_ok}")
    W(f"唯一键数: {n_keys}")
    W(f"因「同键多行」多出来的行数(即重复次数之和): {redundant_by_key}")
    if json_ok:
        W(f"等价: 平均每键出现 {json_ok / max(n_keys, 1):.4f} 次")
    W("")
    W("键出现次数 → 有多少个键落在该桶")
    for bucket in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]:
        if bucket in mult_hist:
            W(f"  {bucket:>3} 次: {mult_hist[bucket]:8d} 个键")
    W("")
    W("— 整行文本完全相同（规范化 strip 换行后）—")
    W(f"至少被复制过 1 次的「不同行内容种类」: {exact_line_dup_strings}")
    W(f"仅出现 1 次的唯一行内容: {exact_line_unique_strings}")
    W(f"因「整行完全相同」多出来的行数: {exact_line_redundant}")
    same_key_diff_content = redundant_by_key - exact_line_redundant
    W("")
    W("解释（粗分）:")
    W(
        f"  · 同键重复里，约有 {max(same_key_diff_content, 0)} 行是「同键但行内容不同」"
        f"（同一函数多条样本、或字段有噪声/微调）。"
    )
    W(
        f"  · 约有 {min(exact_line_redundant, redundant_by_key)} 行与「整行重复」统计重叠；"
        f"若整行重复 ≈ 同键冗余，多为合并/双倍写入。"
    )
    W("")

    # 取出现次数最高的键，拉两条做字段 diff
    top_n = max(0, int(args.top_compare))
    ranked = sorted(key_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    hot_keys = [k for k, c in ranked if c > 1][:top_n]

    if hot_keys and top_n > 0:
        W("=" * 72)
        W(f"以下为出现次数 Top（且次数>1）的前 {len(hot_keys)} 个键，各取前两行比对顶层字段")
        W("")

        # 第二遍扫描：对每个 hot_key 收集至多 2 条解析结果
        need: Dict[FuncKey, int] = {k: 2 for k in hot_keys}
        buf: Dict[FuncKey, List[dict]] = defaultdict(list)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n\r")
                if not s.strip():
                    continue
                try:
                    d = json.loads(s)
                except json.JSONDecodeError:
                    continue
                k = _func_key(d)
                if k not in need:
                    continue
                if len(buf[k]) < need[k]:
                    buf[k].append(d)
                if all(len(buf[x]) >= need[x] for x in hot_keys):
                    break

        for rank, (fk, cnt) in enumerate([(k, key_counts[k]) for k in hot_keys], start=1):
            W(f"--- #{rank}  key={fk!r}  出现行数={cnt} ---")
            recs = buf.get(fk, [])
            if len(recs) < 2:
                W("  (未能扫到 2 条记录，略过对比)")
                W("")
                continue
            a, b = recs[0], recs[1]
            oa, ob, dv = _top_level_diff(a, b)
            if oa:
                W(f"  仅第 1 条有的顶层键: {oa}")
            if ob:
                W(f"  仅第 2 条有的顶层键: {ob}")
            if dv:
                W(f"  同键但值不同（顶层）: {dv}")
                for kk in dv[:12]:
                    la, lb = _stable_json(a.get(kk)), _stable_json(b.get(kk))
                    ha = hashlib.sha256(la.encode("utf-8")).hexdigest()[:12]
                    hb = hashlib.sha256(lb.encode("utf-8")).hexdigest()[:12]
                    W(f"      [{kk}] sha256[:12] {ha} vs {hb}")
                if len(dv) > 12:
                    W(f"      … 其余 {len(dv) - 12} 个字段略")
            else:
                W("  前两条的顶层字段与值完全一致（或仅有嵌套顺序差异已序列化对齐）。")
            W("")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        logger.info("已写入 %s", args.out.resolve())

    logger.info("完成。")


if __name__ == "__main__":
    main()
