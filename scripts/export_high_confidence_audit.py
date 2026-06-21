#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从各项目 binskel jsonl（默认不含 libxml2）中按 validate_sample 筛选「高置信」样本，
每项目导出 K 条人工可读报告（源码 + 反汇编 + M_gt/W），用于目测对齐是否正确。

用法:
  python scripts/export_high_confidence_audit.py
  python scripts/export_high_confidence_audit.py ~/structlift_datasets -o audit.txt
  python scripts/export_high_confidence_audit.py --opt 0 --per-project 3 --min-mean 0.88

依赖: 完整导出需 capstone、pyelftools 与 jsonl 内 ELF 路径可访问；若缺失则自动降级为「仅 M/W + 行号」文本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from validate_alignment import validate_sample  # noqa: E402

try:
    from export_sample_for_manual_check import format_sample_report  # noqa: E402
except ImportError:
    format_sample_report = None  # type: ignore

DEFAULT_PROJECTS = ("zlib", "sqlite", "busybox", "openssl", "coreutils", "curl")


def _format_alignment_light(target: Dict[str, Any]) -> str:
    """无 capstone/elftools 或 ELF 不可读时，仅打印对齐与行号。"""
    fn = target.get("func_name", "?")
    bl = target.get("block_src_lines") or []
    M = target.get("alignment_matrix") or []
    W = target.get("alignment_confidence") or []
    out = [
        f"函数: {fn}",
        f"源: {target.get('source_file')}",
        f"二进制: {target.get('binary')}",
        "\n[block_src_lines]",
    ]
    for i, ln_list in enumerate(bl):
        out.append(f"  BB{i}: {ln_list}")
    out.append("\n[M_gt]")
    for i, row in enumerate(M):
        out.append(f"  BB{i}: {[round(x, 3) for x in row]}")
    out.append("\n[W]")
    for i, row in enumerate(W):
        out.append(f"  BB{i}: {[round(x, 3) for x in row]}")
    out.append(f"\nsignal_stats: {target.get('signal_stats')}")
    return "\n".join(out)


def _format_sample(target: Dict[str, Any]) -> str:
    if format_sample_report is None:
        return _format_alignment_light(target) + "\n\n[提示] 未安装 pyelftools/capstone 或未导入 export_sample，已用轻量导出。\n"
    try:
        return format_sample_report(target)
    except Exception as e:
        return _format_alignment_light(target) + f"\n\n[完整导出失败: {e}]\n"


def _threshold_tiers() -> List[Tuple[float, float, float]]:
    """(min_mean_conf, min_high_conf_ratio, min_bb_coverage) 从严到宽。"""
    return [
        (0.88, 0.90, 0.92),
        (0.85, 0.85, 0.88),
        (0.80, 0.80, 0.85),
        (0.75, 0.70, 0.80),
    ]


def pick_high_confidence(
    jsonl_path: Path,
    k: int,
    tiers: List[Tuple[float, float, float]],
) -> List[Tuple[int, Dict[str, Any], Dict[str, float], str]]:
    """
    返回 [(line_index, sample_dict, metrics, tier_note), ...] 最多 k 条。
    先满足最严 tier 尽量多选；不足则依次用更宽 tier 补满，并避免重复 jsonl 行号。
    """
    used_idx: set = set()
    result: List[Tuple[int, Dict[str, Any], Dict[str, float], str]] = []

    # 预读并缓存全部 (idx,d,r,score)，避免多遍扫描大文件
    rows: List[Tuple[int, Dict[str, Any], Dict[str, float], float]] = []
    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = validate_sample(d)
            score = r["mean_conf"] * 0.6 + r["high_conf_ratio"] * 0.35 + r["coverage"] * 0.05
            rows.append((idx, d, r, score))

    def take_from_tier(min_m: float, min_h: float, min_c: float, note: str) -> None:
        nonlocal result, used_idx
        cand = [
            (sc, idx, d, r)
            for idx, d, r, sc in rows
            if idx not in used_idx
            and r["mean_conf"] >= min_m
            and r["high_conf_ratio"] >= min_h
            and r["coverage"] >= min_c
        ]
        cand.sort(key=lambda x: (-x[0], -x[1]))
        seen_fn: set = set()
        for sc, idx, d, r in cand:
            if len(result) >= k:
                return
            fn = d.get("func_name") or ""
            if fn in seen_fn:
                continue
            seen_fn.add(fn)
            used_idx.add(idx)
            result.append((idx, d, r, note))
        # 去重后仍缺：同名也收
        for sc, idx, d, r in cand:
            if len(result) >= k:
                return
            if idx in used_idx:
                continue
            used_idx.add(idx)
            result.append((idx, d, r, note))

    for min_m, min_h, min_c in tiers:
        if len(result) >= k:
            break
        take_from_tier(
            min_m,
            min_h,
            min_c,
            f"tier mean>={min_m}, high_ratio>={min_h}, cov>={min_c}",
        )

    if len(result) < k:
        flat = [(sc, idx, d, r) for idx, d, r, sc in rows if idx not in used_idx]
        flat.sort(key=lambda x: -x[0])
        note = "fallback: 全文件按 composite score 最高（各 tier 仍不足 k）"
        for sc, idx, d, r in flat:
            if len(result) >= k:
                break
            used_idx.add(idx)
            result.append((idx, d, r, note))

    return result[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description="高置信样本抽审导出（不含 libxml2）")
    ap.add_argument(
        "data_dir",
        nargs="?",
        default=str(Path.home() / "structlift_datasets"),
        help="binskel_*.jsonl 所在目录",
    )
    ap.add_argument("-o", "--output", default=None, help="输出报告路径（默认: data_dir/high_confidence_audit.txt）")
    ap.add_argument("--opt", default="0", help="优化档 o0/o1/o2/o3（默认 o0）")
    ap.add_argument("--per-project", type=int, default=3, help="每项目抽取条数（默认 3）")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help=f"逗号分隔，默认: {','.join(DEFAULT_PROJECTS)}",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    opt = str(args.opt).lstrip("oO") or "0"
    out_path = Path(args.output).expanduser() if args.output else (data_dir / "high_confidence_audit.txt")
    projects = tuple(p.strip() for p in args.projects.split(",") if p.strip())
    k = max(1, args.per_project)
    tiers = _threshold_tiers()

    lines_out: List[str] = []
    lines_out.append("高置信样本对齐抽审（validate_sample 阈值 + 人工导出）")
    lines_out.append(f"目录: {data_dir} | 档位 O{opt} | 每项目 {k} 条 | 项目数 {len(projects)}")
    lines_out.append("筛选: 依次尝试 tier mean_conf / high_conf_ratio / BB_coverage，不足则放宽；仍不足则取全文件 composite score 最高")
    lines_out.append("=" * 100)

    for proj in projects:
        name = f"binskel_{proj}_o{opt}.jsonl"
        jsonl = data_dir / name
        lines_out.append(f"\n\n{'#' * 20} 项目: {proj}  文件: {name} {'#' * 20}\n")
        if not jsonl.is_file():
            lines_out.append(f"  [跳过] 文件不存在: {jsonl}\n")
            continue
        try:
            picked = pick_high_confidence(jsonl, k, tiers)
        except Exception as e:
            lines_out.append(f"  [错误] {e}\n")
            continue
        if not picked:
            lines_out.append("  [无样本]\n")
            continue
        for rank, (idx, d, r, note) in enumerate(picked, 1):
            fn = d.get("func_name", "?")
            lines_out.append(
                f"\n--- [{proj}] #{rank}/{len(picked)}  jsonl行号={idx + 1}(0-based idx={idx})  "
                f"func={fn[:60]} ---\n"
            )
            lines_out.append(
                f"  metrics: mean_conf={r['mean_conf']:.4f}  high_W_ratio={r['high_conf_ratio']:.4f}  "
                f"low_W_ratio={r['low_conf_ratio']:.4f}  BB_cov={r['coverage']:.4f}  "
                f"order={r['order_consistency']:.4f}  cf={r['cf_consistency']:.4f}\n"
            )
            lines_out.append(f"  filter: {note}\n")
            lines_out.append(_format_sample(d))
            lines_out.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines_out)
    out_path.write_text(text, encoding="utf-8")
    print(f"已写入 {len(projects)} 个项目抽审报告 -> {out_path}")
    print(f"若路径在 WSL，请用相同环境下运行；缺 jsonl 的项目已跳过。")


if __name__ == "__main__":
    main()
