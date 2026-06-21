#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFmpeg 四档 binskel（o0–o3）对齐效果统计。

- 与 validate_alignment.py / binskel_alignment_batch_stats.py 同一套 per-sample 指标。
- 单次扫描每文件，同时给出「二进制侧」均值（opcode/BB/边），便于确认各档 ELF 真不同。

用法:
  python3 scripts/stats_ffmpeg_binskel_alignment.py
  python3 scripts/stats_ffmpeg_binskel_alignment.py --dir ~/struclift_wsl/datasets/ffmpeg_binskel
  python3 scripts/stats_ffmpeg_binskel_alignment.py --dir ... --no-md5
  python3 scripts/stats_ffmpeg_binskel_alignment.py --tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_alignment import validate_sample  # noqa: E402


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _pick_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg.expanduser().resolve()
    candidates = [
        Path.home() / "struclift_wsl" / "datasets" / "ffmpeg_binskel",
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "binskel_ffmpeg_o0.jsonl").is_file():
            return c.resolve()
    return candidates[0].resolve()


def aggregate_alignment_and_binary_one_pass(path: Path) -> tuple[dict | None, dict | None]:
    """单次扫描：validate 口径汇总 + 二进制侧均值。"""
    if not path.is_file():
        return None, None
    n = 0
    bad = 0
    sum_cov = sum_mean = sum_high = sum_low = sum_order = sum_cf = 0.0
    has_matrix = 0
    sum_op = sum_bb = sum_edge_pairs = 0.0
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
            op = d.get("bin_opcodes") or []
            bl = d.get("bin_block_lengths") or []
            ei = d.get("bin_edge_index") or [[], []]
            es = ei[0] if len(ei) > 0 else []
            sum_op += len(op)
            sum_bb += len(bl)
            sum_edge_pairs += len(es)
            n += 1
    if n == 0:
        st = {
            "n": 0,
            "bad_lines": bad,
            "cov_pct": None,
            "mean_conf": None,
            "high_pct": None,
            "low_pct": None,
            "order_pct": None,
            "cf_pct": None,
            "m_gt_pct": None,
        }
        bm = {"n": 0, "bad_lines": bad, "mean_opcodes": None, "mean_bb": None, "mean_edges": None}
        return st, bm
    st = {
        "n": n,
        "bad_lines": bad,
        "cov_pct": 100.0 * sum_cov / n,
        "mean_conf": sum_mean / n,
        "high_pct": 100.0 * sum_high / n,
        "low_pct": 100.0 * sum_low / n,
        "order_pct": 100.0 * sum_order / n,
        "cf_pct": 100.0 * sum_cf / n,
        "m_gt_pct": 100.0 * has_matrix / n,
    }
    bm = {
        "n": n,
        "bad_lines": bad,
        "mean_opcodes": sum_op / n,
        "mean_bb": sum_bb / n,
        "mean_edges": sum_edge_pairs / n,
    }
    return st, bm


def main() -> None:
    ap = argparse.ArgumentParser(description="FFmpeg binskel o0–o3 对齐效果统计")
    ap.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="含 binskel_ffmpeg_o0.jsonl … o3.jsonl 的目录",
    )
    ap.add_argument("--no-md5", action="store_true", help="不计算各 jsonl 的 MD5")
    ap.add_argument("--tsv", action="store_true", help="制表符分隔输出")
    args = ap.parse_args()

    d = _pick_dir(args.dir)
    names = [f"binskel_ffmpeg_o{k}.jsonl" for k in range(4)]
    paths = [d / n for n in names]

    rows: list[tuple] = []
    bin_rows: list[tuple] = []
    for name, p in zip(names, paths):
        if p.is_file():
            st, bm = aggregate_alignment_and_binary_one_pass(p)
        else:
            st, bm = None, None
        md5 = None if args.no_md5 else (_file_md5(p) if p.is_file() else None)
        rows.append((name, p, st, md5))
        bin_rows.append((name, p, bm))

    if args.tsv:
        print(
            "文件\tMD5\t样本\t无效行\tBB覆盖率%\t平均置信\t高置信%\t低置信%\t顺序%\t控制流%\tM_gt%\t"
            "Δ均信\tΔ高%%\t均opcode\t均BB\t均边\tΔopcode\tΔBB"
        )
        base_mean = base_high = base_op = base_bb = None
        for (name, p, st, md5), (_, _, bm) in zip(rows, bin_rows):
            md5s = (md5 or "") if not args.no_md5 else ""
            if st is None or st["n"] == 0:
                print(f"{name}\t{md5s}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-")
                continue
            if name.endswith("_o0.jsonl"):
                base_mean, base_high = st["mean_conf"], st["high_pct"]
                base_op = bm["mean_opcodes"] if bm and bm.get("mean_opcodes") is not None else None
                base_bb = bm["mean_bb"] if bm and bm.get("mean_bb") is not None else None
            d_m = d_h = d_op = d_bb = "-"
            if base_mean is not None and not name.endswith("_o0.jsonl"):
                if st["mean_conf"] is not None:
                    d_m = f"{st['mean_conf'] - base_mean:+.4f}"
                if st["high_pct"] is not None and base_high is not None:
                    d_h = f"{st['high_pct'] - base_high:+.2f}"
                if base_op is not None and bm and bm.get("mean_opcodes") is not None:
                    d_op = f"{bm['mean_opcodes'] - base_op:+.2f}"
                if base_bb is not None and bm and bm.get("mean_bb") is not None:
                    d_bb = f"{bm['mean_bb'] - base_bb:+.2f}"
            op_s = f"{bm['mean_opcodes']:.2f}" if bm and bm.get("mean_opcodes") is not None else "-"
            bb_s = f"{bm['mean_bb']:.2f}" if bm and bm.get("mean_bb") is not None else "-"
            eg_s = f"{bm['mean_edges']:.2f}" if bm and bm.get("mean_edges") is not None else "-"
            print(
                f"{name}\t{md5s}\t{st['n']}\t{st['bad_lines']}\t"
                f"{st['cov_pct']:.2f}\t{st['mean_conf']:.4f}\t{st['high_pct']:.2f}\t"
                f"{st['low_pct']:.2f}\t{st['order_pct']:.2f}\t{st['cf_pct']:.2f}\t"
                f"{st['m_gt_pct']:.2f}\t{d_m}\t{d_h}\t{op_s}\t{bb_s}\t{eg_s}\t{d_op}\t{d_bb}"
            )
        return

    print(f"FFmpeg binskel 对齐统计  目录: {d}\n")
    if not d.is_dir():
        print("错误: 目录不存在", file=sys.stderr)
        sys.exit(1)

    if not args.no_md5:
        print("各 jsonl MD5（四档应两两不同；若相同说明仍是同一份数据）:")
        for name, p, st, md5 in rows:
            if md5:
                print(f"  {md5}  {name}")
            elif p.is_file():
                print(f"  (跳过)  {name}")
            else:
                print(f"  缺失    {name}")
        print()

    print("### 对齐质量（与 validate_alignment.py 一致）\n")
    print(
        "| 档位 | 样本 | 无效行 | BB覆盖率% | 平均置信 | 高置信% | 低置信% | 顺序% | 控制流% | M_gt% | Δ均信 | Δ高% |"
    )
    print(
        "|------|------|--------|-----------|----------|---------|---------|-------|---------|-------|-------|------|"
    )

    base_mean = base_high = None
    for name, p, st, md5 in rows:
        if st is None:
            print(f"| {name} | 缺失 | - | - | - | - | - | - | - | - | - | - |")
            continue
        if st["n"] == 0:
            print(f"| {name} | 0 | {st['bad_lines']} | - | - | - | - | - | - | - | - | - |")
            continue
        if name.endswith("_o0.jsonl"):
            base_mean, base_high = st["mean_conf"], st["high_pct"]
        d_m = d_h = "-"
        if base_mean is not None and not name.endswith("_o0.jsonl"):
            if st["mean_conf"] is not None:
                d_m = f"{st['mean_conf'] - base_mean:+.4f}"
            if st["high_pct"] is not None and base_high is not None:
                d_h = f"{st['high_pct'] - base_high:+.2f}"
        print(
            f"| {name} | {st['n']} | {st['bad_lines']} | {st['cov_pct']:.2f} | {st['mean_conf']:.4f} | "
            f"{st['high_pct']:.2f} | {st['low_pct']:.2f} | {st['order_pct']:.2f} | {st['cf_pct']:.2f} | "
            f"{st['m_gt_pct']:.2f} | {d_m} | {d_h} |"
        )

    print("\n### 二进制侧（validate 未读 opcode；各档应随 -O 变化）\n")
    print("| 档位 | 均opcode/样本 | 均BB数 | 均CFG边 | Δopcode(vsO0) | ΔBB(vsO0) |")
    print("|------|----------------|--------|---------|----------------|------------|")

    base_op = base_bb = None
    for (name, p, st, md5), (_, _, bm) in zip(rows, bin_rows):
        if bm is None or bm["n"] == 0:
            print(f"| {name} | - | - | - | - | - |")
            continue
        if name.endswith("_o0.jsonl"):
            base_op, base_bb = bm["mean_opcodes"], bm["mean_bb"]
        d_op = d_bb = "-"
        if base_op is not None and not name.endswith("_o0.jsonl"):
            d_op = f"{bm['mean_opcodes'] - base_op:+.2f}"
            d_bb = f"{bm['mean_bb'] - base_bb:+.2f}"
        print(
            f"| {name} | {bm['mean_opcodes']:.2f} | {bm['mean_bb']:.2f} | {bm['mean_edges']:.2f} | "
            f"{d_op} | {d_bb} |"
        )

    print(
        "\n说明: Δ 列为相对 O0；对齐表与 scripts/validate_alignment.py 一致；"
        "修复 compile_ffmpeg 后四档 MD5 应不同，且「均 opcode/BB」通常应有差异。"
    )


if __name__ == "__main__":
    main()
