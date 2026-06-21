#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9 项目 × 5 档 = 45 个 binskel jsonl：全量逐样本、逐基本块统计

  - total_bb: 全部基本块条数（所有样本的 n_bb 之和）
  - bb_has_dwarf: block_src_lines[i] 非空（列表长度 > 0）
  - bb_has_m: alignment_matrix 第 i 行存在 >0 的元素
  - bb_both: 同时满足以上两者

用法:
  python scripts/stats_45_bb_dwarf_m_full.py --datasets E:/structlift_datasets \\
    -o E:/structlift_datasets/bb_dwarf_m_stats_45.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from binskel_alignment_batch_stats import DEFAULT_PROJECTS  # noqa: E402


def jsonl_name(proj: str, opt: str) -> str:
    o = opt.strip().lower()
    if o == "os":
        return f"binskel_{proj}_os.jsonl"
    return f"binskel_{proj}_o{o}.jsonl"


def count_sample_bb(d: dict) -> tuple[int, int, int, int, int]:
    """
    返回 (n_bb, has_dwarf, has_m, both, mismatch_len)
    n_bb = len(block_src_lines)；矩阵行不足处视为无 M。
    mismatch: len(bl) != len(M) 时样本级 +1。
    """
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    if not isinstance(bl, list):
        bl = []
    if not isinstance(M, list):
        M = []

    n_b = len(bl)
    mismatch = 1 if (M and n_b != len(M)) else 0

    has_dwarf = 0
    has_m = 0
    both = 0
    for i in range(n_b):
        row = M[i] if i < len(M) else []
        hd = isinstance(bl[i], list) and len(bl[i]) > 0
        m_ok = bool(row) and any(float(x) > 1e-12 for x in row)
        if hd:
            has_dwarf += 1
        if m_ok:
            has_m += 1
        if hd and m_ok:
            both += 1

    return n_b, has_dwarf, has_m, both, mismatch


def scan_file(path: Path) -> tuple[int, int, int, int, int, int, int]:
    """total_bb, dwarf, m, both, n_samples, n_mismatch_samples, bad_lines"""
    tot_b = tot_d = tot_m = tot_both = 0
    n_samp = 0
    n_mis = 0
    bad = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            n_samp += 1
            tb, d_, m_, b_, mis = count_sample_bb(d)
            tot_b += tb
            tot_d += d_
            tot_m += m_
            tot_both += b_
            n_mis += mis
    return tot_b, tot_d, tot_m, tot_both, n_samp, n_mis, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="45 jsonl 全量 BB×DWARF×M 统计")
    ap.add_argument("--datasets", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help="逗号分隔，默认 9 项目",
    )
    ap.add_argument("--opt-levels", default="0,1,2,3,os")
    ap.add_argument("-o", "--output", type=Path, default=None, help="TSV 输出路径")
    args = ap.parse_args()

    base = args.datasets.expanduser().resolve()
    projects = [x.strip() for x in args.projects.split(",") if x.strip()]
    opts = [x.strip().lower() for x in args.opt_levels.split(",") if x.strip()]

    rows: list[tuple] = []
    G = [0, 0, 0, 0, 0, 0, 0]  # tb, d, m, both, samp, mis, bad

    for proj in projects:
        for ot in opts:
            name = jsonl_name(proj, ot)
            path = base / name
            label = f"{proj}_{ot}"
            if not path.is_file():
                if ot == "os":
                    alt_n = f"binskel_{proj}_os_multibin.jsonl"
                else:
                    alt_n = f"binskel_{proj}_o{ot}_multibin.jsonl"
                alt_p = base / alt_n
                if alt_p.is_file():
                    path = alt_p
                    name = alt_n
            if not path.is_file():
                rows.append((label, name, 0, 0, 0, 0, 0, 0, 0, "MISSING"))
                continue
            tb, td, tm, tboth, ns, nmis, bad = scan_file(path)
            rows.append(
                (
                    label,
                    name,
                    tb,
                    td,
                    tm,
                    tboth,
                    ns,
                    nmis,
                    bad,
                    "OK",
                )
            )
            G[0] += tb
            G[1] += td
            G[2] += tm
            G[3] += tboth
            G[4] += ns
            G[5] += nmis
            G[6] += bad

    lines = []
    hdr = (
        "dataset\tjsonl\ttotal_bb\tbb_has_dwarf\tbb_has_M\tbb_both\t"
        "pct_dwarf\tpct_M\tpct_both\tn_samples\tn_mismatch_matrix_len\tbad_json\tstatus"
    )
    lines.append(hdr)

    def pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:.4f}" if b else "-"

    for r in rows:
        label, name, tb, td, tm, tboth, ns, nmis, bad, st = r
        lines.append(
            f"{label}\t{name}\t{tb}\t{td}\t{tm}\t{tboth}\t"
            f"{pct(td, tb)}\t{pct(tm, tb)}\t{pct(tboth, tb)}\t{ns}\t{nmis}\t{bad}\t{st}"
        )

    tb, td, tm, tboth, ns, nmis, bad = G
    lines.append(
        f"TOTAL\t-\t{tb}\t{td}\t{tm}\t{tboth}\t"
        f"{pct(td, tb)}\t{pct(tm, tb)}\t{pct(tboth, tb)}\t{ns}\t{nmis}\t{bad}\t-"
    )

    out_txt = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_txt, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    print(out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
