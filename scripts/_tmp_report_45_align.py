#!/usr/bin/env python3
"""One-shot: 9 proj x 5 opt (O0-O3+Os) alignment table from a single jsonl root."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import json

from binskel_alignment_batch_stats import DEFAULT_PROJECTS  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402


def aggregate_jsonl_fast(path: Path) -> dict | None:
    """与 binskel_alignment_batch_stats.aggregate_jsonl 同源，但不做 deep_validate（大文件可快几个数量级）。"""
    if not path.is_file():
        return None
    n = 0
    sum_cov = sum_mean = sum_high = sum_low = sum_order = sum_cf = 0.0
    has_matrix = 0
    bad = 0
    n_stmt_tier = n_line_tier = n_tier_unknown = 0
    sum_bb_cov = 0.0
    n_bb_cov = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            M = d.get("alignment_matrix")
            if isinstance(M, list) and M and isinstance(M[0], list) and len(M[0]) > 0:
                has_matrix += 1
            r = validate_sample(d)
            sum_cov += r["coverage"]
            sum_mean += r["mean_conf"]
            sum_high += r["high_conf_ratio"]
            sum_low += r["low_conf_ratio"]
            sum_order += r["order_consistency"]
            sum_cf += r["cf_consistency"]
            tier = d.get("src_encoding_tier")
            if tier == "stmt":
                n_stmt_tier += 1
            elif tier == "line_fallback":
                n_line_tier += 1
            else:
                n_tier_unknown += 1
            bc = d.get("bb_alignment_coverage")
            if isinstance(bc, (int, float)):
                sum_bb_cov += float(bc)
                n_bb_cov += 1
            n += 1
    if n == 0:
        return {
            "n": 0,
            "bad_lines": bad,
            "cov_pct": None,
            "mean_conf": None,
            "high_pct": None,
            "low_pct": None,
            "order_pct": None,
            "cf_pct": None,
            "m_gt_pct": None,
            "mean_bb_cov_pct": None,
            "stmt_tier_pct": None,
            "line_tier_pct": None,
            "tier_unknown_pct": None,
            "matrix_consistent_pct": None,
            "n_inconsistent": 0,
        }
    return {
        "n": n,
        "bad_lines": bad,
        "cov_pct": 100.0 * sum_cov / n,
        "mean_conf": sum_mean / n,
        "high_pct": 100.0 * sum_high / n,
        "low_pct": 100.0 * sum_low / n,
        "order_pct": 100.0 * sum_order / n,
        "cf_pct": 100.0 * sum_cf / n,
        "m_gt_pct": 100.0 * has_matrix / n,
        "mean_bb_cov_pct": (100.0 * sum_bb_cov / n_bb_cov) if n_bb_cov else None,
        "stmt_tier_pct": 100.0 * n_stmt_tier / n,
        "line_tier_pct": 100.0 * n_line_tier / n,
        "tier_unknown_pct": 100.0 * n_tier_unknown / n,
        "matrix_consistent_pct": None,
        "n_inconsistent": 0,
    }


def main() -> None:
    root = Path(r"e:/structlift_datasets")
    opts = [
        ("O0", lambda p: f"binskel_{p}_o0.jsonl"),
        ("O1", lambda p: f"binskel_{p}_o1.jsonl"),
        ("O2", lambda p: f"binskel_{p}_o2.jsonl"),
        ("O3", lambda p: f"binskel_{p}_o3.jsonl"),
        ("Os", lambda p: f"binskel_{p}_os.jsonl"),
    ]
    rows = []
    for proj in DEFAULT_PROJECTS:
        for label, fnl in opts:
            fn = fnl(proj)
            p = root / fn
            st = aggregate_jsonl_fast(p) if p.is_file() else None
            rows.append((proj, label, fn, st))

    print(
        "| 项目 | 档位 | jsonl | 样本n | BB覆盖率% | 平均置信 | 高置信% | 低置信% | "
        "顺序% | 控制流% | M_gt% | bb_cov均% | stmt档% |"
    )
    print(
        "|---|---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for proj, label, fn, st in rows:
        if st is None:
            print(f"| {proj} | {label} | `{fn}` | **缺失** | - | - | - | - | - | - | - | - |")
            continue
        if st["n"] == 0:
            bad = st.get("bad_lines", 0)
            print(
                f"| {proj} | {label} | `{fn}` | 0 | - | - | - | - | - | - | - | - | "
                f"*(坏行 {bad})*"
            )
            continue
        bb = st.get("mean_bb_cov_pct")
        bb_s = f"{bb:.2f}" if bb is not None else "-"
        print(
            f"| {proj} | {label} | `{fn}` | {st['n']} | {st['cov_pct']:.2f} | "
            f"{st['mean_conf']:.4f} | {st['high_pct']:.2f} | {st['low_pct']:.2f} | "
            f"{st['order_pct']:.2f} | {st['cf_pct']:.2f} | {st['m_gt_pct']:.2f} | "
            f"{bb_s} | {st['stmt_tier_pct']:.2f} |"
        )

    ok = sum(1 for *_, st in rows if st and st["n"] > 0)
    print()
    print(
        f"有效数据集: **{ok}** / **{len(rows)}**（目录 `{root}`）。"
        "*M↔行自洽%* 需全量 `deep_validate`，本表为快速统计已省略。"
    )


if __name__ == "__main__":
    main()
