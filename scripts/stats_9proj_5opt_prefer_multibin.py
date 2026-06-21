#!/usr/bin/env python3
"""9 项目 × 5 档；coreutils/ffmpeg/openssh 优先 multibin jsonl，其余标准名。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from binskel_alignment_batch_stats import DEFAULT_PROJECTS, aggregate_jsonl  # noqa: E402

MULTIBIN = frozenset({"coreutils", "ffmpeg", "openssh"})


def pick_path(proj: str, opt: str, base: Path) -> tuple[Path | None, str]:
    if proj in MULTIBIN:
        if opt == "os":
            cand = base / f"binskel_{proj}_os_multibin.jsonl"
        else:
            cand = base / f"binskel_{proj}_o{opt}_multibin.jsonl"
        if cand.is_file():
            return cand, cand.name
    if opt == "os":
        cand = base / f"binskel_{proj}_os.jsonl"
    else:
        cand = base / f"binskel_{proj}_o{opt}.jsonl"
    if cand.is_file():
        return cand, cand.name
    return None, cand.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path.home() / "structlift_datasets")
    args = ap.parse_args()
    base = args.datasets.expanduser().resolve()

    opts = ["0", "1", "2", "3", "os"]
    rows: list[dict] = []
    total_n = 0
    w_cov = w_mc = w_high = w_order = w_cf = 0.0
    w_mgt = 0.0
    missing: list[str] = []

    for proj in DEFAULT_PROJECTS:
        for opt in opts:
            path, fname = pick_path(proj, opt, base)
            label = f"{proj}_{opt}"
            if path is None:
                missing.append(f"{label} ({fname})")
                rows.append({"label": label, "file": fname, "n": 0, "path": None})
                continue
            st = aggregate_jsonl(path)
            if not st or st["n"] == 0:
                missing.append(f"{label} empty")
                rows.append({"label": label, "file": fname, "n": 0, "path": path})
                continue
            n = st["n"]
            total_n += n
            w_cov += (st["cov_pct"] or 0) * n
            w_mc += (st["mean_conf"] or 0) * n
            w_high += (st["high_pct"] or 0) * n
            w_order += (st["order_pct"] or 0) * n
            w_cf += (st["cf_pct"] or 0) * n
            w_mgt += (st["m_gt_pct"] or 0) * n
            rows.append(
                {
                    "label": label,
                    "file": path.name,
                    "n": n,
                    "cov": st["cov_pct"],
                    "mean_conf": st["mean_conf"],
                    "high": st["high_pct"],
                    "order": st["order_pct"],
                    "cf": st["cf_pct"],
                    "m_gt": st["m_gt_pct"],
                    "stmt": st["stmt_tier_pct"],
                    "path": path,
                }
            )

    print("# 9 项目 × 5 档（coreutils/ffmpeg/openssh 优先 *_multibin.jsonl）")
    print(f"# 根目录: {base}\n")

    print("| 槽位 | 实际文件 | 样本数 | BB覆盖% | 均信 | 高信% | 顺序% | 控制流% | M_gt% | stmt档% |")
    print("|------|----------|--------|---------|------|-------|-------|---------|-------|---------|")
    for r in rows:
        if r["n"] == 0:
            print(f"| {r['label']} | {r['file']} | — | — | — | — | — | — | — | — |")
            continue
        print(
            f"| {r['label']} | `{r['file']}` | {r['n']} | {r['cov']:.2f} | {r['mean_conf']:.4f} | "
            f"{r['high']:.2f} | {r['order']:.2f} | {r['cf']:.2f} | {r['m_gt']:.2f} | {r['stmt']:.2f} |"
        )

    print()
    if total_n:
        print("## 汇总（按样本数加权平均）")
        print(f"- **总样本行数**（45 个 jsonl 相加）: **{total_n:,}**")
        print(f"- **加权 BB 覆盖率**: {w_cov / total_n:.2f}%")
        print(f"- **加权 平均置信度**: {w_mc / total_n:.4f}")
        print(f"- **加权 高置信占比**: {w_high / total_n:.2f}%")
        print(f"- **加权 顺序一致性**: {w_order / total_n:.2f}%")
        print(f"- **加权 控制流一致性**: {w_cf / total_n:.2f}%")
        print(f"- **加权 含 M_gt 比例**: {w_mgt / total_n:.2f}%")

    def _lab_proj(lab: str) -> str:
        for pr in sorted(DEFAULT_PROJECTS, key=len, reverse=True):
            if lab.startswith(pr + "_"):
                return pr
        return lab.split("_", 1)[0]

    by_proj: dict[str, int] = {}
    for r in rows:
        if r["n"] == 0:
            continue
        p = _lab_proj(r["label"])
        by_proj[p] = by_proj.get(p, 0) + r["n"]

    print("\n## 各项目样本合计（五档相加）")
    for p in DEFAULT_PROJECTS:
        print(f"- **{p}**: {by_proj.get(p, 0):,}")

    if missing:
        print(f"\n## 缺失或空（{len(missing)}）")
        for m in missing[:20]:
            print(f"- {m}")
        if len(missing) > 20:
            print(f"- … 共 {len(missing)} 条")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
