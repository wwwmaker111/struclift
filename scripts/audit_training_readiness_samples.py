#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9 项目 × 5 档（O0–O3 + Os）共 45 个 binskel 数据集：每个数据集抽取若干条「高置信 + 足够复杂」样本，做：
  - 结构分析：deep_validate.analyze_sample
  - 语义自洽：check_alignment_semantics.check_sample

默认：每数据集 3 条（45×3=135 条）；`--opt-levels` / `--per-dataset` 可调。

用法:
  python3 scripts/audit_training_readiness_samples.py --all-default-paths --preset-src-mnt-e \\
    --opt-levels 0,1,2,3,os --per-dataset 3 \\
    --output /mnt/e/structlift/training_readiness_audit_9x5x3.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from binskel_alignment_batch_stats import (  # noqa: E402
    DEFAULT_PROJECTS,
    dedupe_paths,
    locate_jsonl,
    default_multi_root_search_list,
)
from deep_validate import analyze_sample  # noqa: E402
from export_high_confidence_audit import (  # noqa: E402
    _threshold_tiers,
    pick_high_confidence,
)
from validate_alignment import validate_sample  # noqa: E402

try:
    from check_alignment_semantics import check_sample as semantic_check_sample  # noqa: E402
except ImportError:
    semantic_check_sample = None  # type: ignore


def _parse_src_map(pairs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        out[k] = Path(v).expanduser().resolve()
    return out


def _preset_mnt_e() -> Dict[str, Path]:
    """常见 /mnt/e 布局；目录不存在则跳过。"""
    raw = {
        "zlib": "/mnt/e/structlift/zlib_build",
        "curl": "/mnt/e/structlift_src_datasets/curl/curl-8.7.1",
        "sqlite": "/mnt/e/structlift_src_datasets/sqlite/sqlite-autoconf-3520000",
        "busybox": "/mnt/e/structlift_src_datasets/busybox/busybox-1.36.1",
        "openssl": "/mnt/e/structlift_src_datasets/openssl/openssl-3.3.1",
        "openssh": "/mnt/e/structlift_src_datasets/openssh/openssh-9.8p1",
        "libxml2": "/mnt/e/structlift_src_datasets/libxml2/libxml2-2.12.7",
        "ffmpeg": "/mnt/e/structlift_src_datasets/ffmpeg/ffmpeg-7.1",
        "coreutils": "/mnt/e/structlift_src_datasets/coreutils/coreutils-9.4",
    }
    out: Dict[str, Path] = {}
    for k, v in raw.items():
        pp = Path(v)
        if pp.is_dir():
            out[k] = pp.resolve()
    return out


def binskel_jsonl_name(proj: str, opt_token: str) -> str:
    """opt_token: 0–3 或 os/Os/OS → binskel_<proj>_oN.jsonl / _os.jsonl"""
    t = opt_token.strip().lower()
    if t == "os":
        return f"binskel_{proj}_os.jsonl"
    return f"binskel_{proj}_o{t}.jsonl"


def parse_opt_levels(s: str) -> List[str]:
    """逗号分隔，如 0,1,2,3,os → ['0','1','2','3','os']"""
    out: List[str] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p.lower() if p.lower() in ("os",) else p.lstrip("oO") or "0")
    return out if out else ["0"]


def _verdict_struct(v: Dict[str, Any], r_val: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    grade = "OK"
    if not v.get("matrix_lines_consistent", True):
        grade = "FAIL"
        notes.append("M_gt 与 block_src_lines 不自洽")
    if v.get("empty_ratio", 0) > 0.15:
        grade = "WARN" if grade == "OK" else grade
        notes.append(f"空对齐 BB 比例偏高: {v['empty_ratio']:.1%}")
    if v.get("order_consistency", 1) < 0.75:
        grade = "WARN" if grade == "OK" else grade
        notes.append(f"顺序一致性偏低: {v['order_consistency']:.2f}")
    if len(v.get("jumps") or []) > max(3, v.get("n_blocks", 0) // 50):
        grade = "WARN" if grade == "OK" else grade
        notes.append(f"语句序跳跃处数: {len(v['jumps'])}")
    if r_val.get("coverage", 0) < 0.85:
        grade = "WARN" if grade == "OK" else grade
        notes.append(f"validate BB 覆盖率: {r_val['coverage']:.3f}")
    if r_val.get("mean_conf", 0) < 0.82:
        grade = "WARN" if grade == "OK" else grade
        notes.append(f"validate 平均置信: {r_val['mean_conf']:.3f}")
    return grade, notes


def _verdict_sem(sem: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not sem:
        return "SKIP", ["未配置源码根或未安装语义检查"]
    if sem.get("skip"):
        return "SKIP", [sem.get("reason", "skip")]
    notes: List[str] = []
    grade = "OK"
    if sem.get("n_viol_peak", 0) > 0:
        grade = "WARN"
        notes.append(f"主峰与 tree-sitter 语句/DWARF 无交: {sem['n_viol_peak']} 个 BB")
    if sem.get("n_viol_highw", 0) > 0:
        grade = "WARN"
        notes.append(f"高W 与 DWARF 无交: {sem['n_viol_highw']} 处")
    if sem.get("weak_map_cols", 0) > max(2, (sem.get("n_col") or 0) // 20):
        grade = "WARN" if grade == "OK" else grade
        notes.append(f"弱列映射列数: {sem['weak_map_cols']}/{sem.get('n_col')}")
    return grade, notes


def _format_bb_level_detail(
    d: Dict[str, Any],
    v: Dict[str, Any],
    sem: Optional[Dict[str, Any]],
    max_preview: int = 14,
) -> str:
    """
    对齐粒度是基本块；函数只是容器。输出：BB 总数、语义冲突 BB 索引、前若干 BB 的 dwarf→主峰列。
    """
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    sl = d.get("source_lines") or []
    n_bb = len(bl)
    bb_pri: List[int] = list(v.get("bb_primary") or [])
    aligned = 0
    for i in range(n_bb):
        if i < len(M) and M[i] and any(float(x) > 1e-6 for x in M[i]):
            aligned += 1
    parts: List[str] = []
    parts.append(
        f"- **BB 级说明**: 训练/对齐标签是 **每个基本块** 与语句列的对应；下表为函数内 BB 汇总。\n"
    )
    cov = float(v.get("coverage") or 0)
    parts.append(
        f"  - 总 BB={n_bb}；矩阵非空对齐 BB={aligned}；"
        f"BB 覆盖率（analyze_sample）={cov:.2%}\n"
    )
    if sem and not sem.get("skip"):
        parts.append(
            f"  - 语义脚本逐 BB 检查数 n_bb_checked={sem.get('n_bb_checked', '-')}"
            f"（有 DWARF 行且矩阵行非零的 BB）\n"
        )
        vp = sem.get("viol_peak") or []
        vh = sem.get("viol_highw") or []
        if vp:
            bbs = [str(x.get("bb")) for x in vp]
            tail = "…" if len(bbs) > 24 else ""
            parts.append(
                f"  - **主峰启发式可疑 BB 索引**（共 {len(vp)} 个）: {', '.join(bbs[:24])}{tail}\n"
            )
        if vh:
            bbs = [str(x.get("bb")) for x in vh]
            parts.append(
                f"  - **高W 启发式可疑 (BB,j)**（共 {len(vh)} 处，列前 12）: "
                f"{[(x.get('bb'), x.get('j')) for x in vh[:12]]}\n"
            )
    elif sem and sem.get("skip"):
        parts.append(f"  - 语义未跑 BB 级对照: {sem.get('reason')}\n")

    n_show = min(max_preview, n_bb)
    if n_show > 0:
        parts.append(f"  - 前 **{n_show}** 个 BB：DWARF 行 → 主峰列 j（及 source_lines[j] 行号）\n")
        for i in range(n_show):
            lines_dw = bl[i] if i < len(bl) else []
            j = bb_pri[i] if i < len(bb_pri) else -1
            stmt_ln = sl[j] if j >= 0 and j < len(sl) else None
            m_peak = 0.0
            if i < len(M) and M[i] and j >= 0 and j < len(M[i]):
                m_peak = float(M[i][j])
            dw_s = str(lines_dw[:8]) + ("…" if len(lines_dw) > 8 else "")
            parts.append(
                f"    - `BB{i}`: dwarf {dw_s} → `j={j}`"
                f"{f', stmt_line≈{stmt_ln}' if stmt_ln is not None else ''}, M_peak={m_peak:.3f}\n"
            )
        if n_bb > n_show:
            parts.append(f"    - … 共 {n_bb} 个 BB，此处仅预览前 {n_show} 个\n")
    return "".join(parts)


def _training_suitability(g_struct: str, g_sem: str) -> str:
    if g_struct == "FAIL":
        return "不建议（结构自洽失败）"
    if g_struct == "WARN" and g_sem == "WARN":
        return "谨慎使用（结构+语义均有告警）"
    if g_sem == "WARN":
        return "谨慎使用（语义自洽告警）"
    if g_struct == "WARN":
        return "可用但建议复核（结构告警）"
    if g_sem == "SKIP":
        return "结构可接受；语义未检，大项目建议补源码路径后重跑"
    return "可作为高置信训练样本（在抽查前提下）"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="高置信 binskel 样本：结构 + 语义审计，评估训练可用性"
    )
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--output", "-o", type=Path, default=None, help="Markdown 报告路径")
    ap.add_argument(
        "--opt-levels",
        default="0,1,2,3,os",
        help="逗号分隔：0–3 与 os（默认 45 数据集：O0–O3 + Os）",
    )
    ap.add_argument(
        "--opt",
        default=None,
        help="已弃用：请用 --opt-levels；若仍传入则等价于单档",
    )
    ap.add_argument(
        "--per-dataset",
        type=int,
        default=3,
        help="每个 binskel jsonl 抽取条数（默认 3，即 45×3=135 条）",
    )
    ap.add_argument(
        "--per-project",
        type=int,
        default=None,
        help="同 --per-dataset（兼容旧参数）",
    )
    ap.add_argument("--min-blocks", type=int, default=5)
    ap.add_argument("--min-used-cols", type=int, default=3)
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help=f"逗号分隔，默认 9 项: {','.join(DEFAULT_PROJECTS)}",
    )
    ap.add_argument(
        "--src",
        action="append",
        default=[],
        metavar="PROJ=PATH",
        help="源码根（与构建 --src 一致；可重复）",
    )
    ap.add_argument(
        "--preset-src-mnt-e",
        action="store_true",
        help="尝试填充 /mnt/e 下常见源码路径（存在才用）",
    )
    ap.add_argument(
        "--bb-preview",
        type=int,
        default=14,
        help="每条样本在报告中列出前多少个 BB 的 dwarf→主峰摘要（默认 14）",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if args.all_default_paths:
        search_dirs = default_multi_root_search_list(out_dir)
        for p in args.search_dir:
            search_dirs.extend([Path(p).expanduser().resolve()])
        search_dirs = dedupe_paths(search_dirs)
    else:
        search_dirs = [out_dir] + [Path(p).expanduser().resolve() for p in args.search_dir]

    src_map = _parse_src_map(args.src)
    if args.preset_src_mnt_e:
        for k, v in _preset_mnt_e().items():
            src_map.setdefault(k, v)

    if args.opt is not None:
        opt_levels = parse_opt_levels(str(args.opt))
    else:
        opt_levels = parse_opt_levels(args.opt_levels)

    projects = [x.strip() for x in args.projects.split(",") if x.strip()]
    k = args.per_dataset if args.per_project is None else args.per_project
    k = max(1, int(k))
    tiers = _threshold_tiers()

    out_path = args.output
    if out_path is None:
        out_path = out_dir / "training_readiness_audit_9x5x3.md"

    n_ds = len(projects) * len(opt_levels)
    lines: List[str] = []
    lines.append("# 高置信 binskel 样本：结构 + 语义审计（训练可用性）\n")
    lines.append("自动化结论：**不替代**人工对照反汇编；用于筛掉明显不自洽或与语句语义冲突的样本。\n")
    lines.append("## 抽样规则\n")
    lines.append(
        f"- **{len(projects)}** 项目 × **{len(opt_levels)}** 档 = **{n_ds}** 个数据集；每数据集 **{k}** 条；"
        f"至多 **{n_ds * k}** 条样本。\n"
    )
    lines.append(
        f"- `pick_high_confidence`：高 mean_conf / high_W / BB 覆盖率，"
        f"且 **BB>={args.min_blocks}**、**矩阵有效列>={args.min_used_cols}**。\n"
    )
    lines.append(f"- 优化档: **{','.join(opt_levels)}**；搜索路径: `{search_dirs[0]}` 等。\n")

    n_missing = 0
    n_empty_pick = 0
    n_rows = 0

    for proj in projects:
        for opt_tok in opt_levels:
            name = binskel_jsonl_name(proj, opt_tok)
            path_used, root_used = locate_jsonl(name, search_dirs)
            opt_label = "Os" if opt_tok.lower() == "os" else f"O{opt_tok}"
            lines.append(f"\n### `{proj}` / {opt_label} — `{name}`\n")
            if not path_used:
                lines.append(f"**缺失文件**（已跳过）\n")
                n_missing += 1
                continue
            lines.append(f"- 数据文件: `{path_used}`\n")
            if root_used:
                lines.append(f"- 解析目录: `{root_used}`\n")
            lines.append(
                "\n| 函数 | 结构 | 语义 | 训练适宜性 | mean_conf | BB覆盖率 | 单点对齐% | 语义冲突(峰/高W) |\n"
                "|------|------|------|------------|-----------|----------|-----------|-------------------|"
            )

            try:
                picked = pick_high_confidence(
                    path_used,
                    k,
                    tiers,
                    min_blocks=max(1, args.min_blocks),
                    min_used_cols=max(1, args.min_used_cols),
                )
            except Exception as e:
                lines.append(f"**错误**: {e}\n")
                continue

            if not picked:
                lines.append("_无满足阈值的样本（可放宽 --min-blocks / --min-used-cols）_\n")
                n_empty_pick += 1
                continue

            src_root = src_map.get(proj)

            for rank, (idx, d, r_val, note) in enumerate(picked, 1):
                n_rows += 1
                fn = d.get("func_name", "?")
                v = analyze_sample(d)
                g_st, n_st = _verdict_struct(v, r_val)

                sem: Optional[Dict[str, Any]] = None
                if semantic_check_sample is not None and src_root and src_root.is_dir():
                    try:
                        sem = semantic_check_sample(d, src_root, high_w=0.85)
                    except Exception as e:
                        sem = {"skip": True, "reason": f"semantic_check 异常: {e}"}
                elif not src_root:
                    sem = None
                else:
                    sem = {"skip": True, "reason": f"源码根不存在: {src_root}"}

                g_se, n_se = _verdict_sem(sem)
                suit = _training_suitability(g_st, g_se)

                viol_p = sem.get("n_viol_peak", "-") if sem and not sem.get("skip") else "-"
                viol_w = sem.get("n_viol_highw", "-") if sem and not sem.get("skip") else "-"
                viol_cell = f"{viol_p} / {viol_w}"

                lines.append(
                    f"| `{fn[:56]}` | {g_st} | {g_se} | {suit} | "
                    f"{r_val['mean_conf']:.3f} | {r_val['coverage']:.3f} | {v['onehot_ratio']:.2f} | {viol_cell} |"
                )

                lines.append(
                    f"\n<details><summary>详情 {proj} {opt_label} #{rank} <code>{fn}</code></summary>\n\n"
                )
                lines.append(f"- 筛选说明: `{note}`\n")
                lines.append(
                    f"- validate: mean_conf={r_val['mean_conf']:.4f}, high_W%={r_val['high_conf_ratio']:.4f}, "
                    f"coverage={r_val['coverage']:.4f}, order={r_val['order_consistency']:.4f}, cf={r_val['cf_consistency']:.4f}\n"
                )
                lines.append(
                    f"- 结构: n_bb={v['n_blocks']}, n_stmt={v['n_stmts']}, matrix_lines_consistent={v['matrix_lines_consistent']}, "
                    f"onehot={v['onehot_ratio']:.2%}, multi={v['multi_ratio']:.2%}, empty={v['empty_ratio']:.2%}, "
                    f"order_consistency={v['order_consistency']:.2%}, jumps={len(v.get('jumps') or [])}\n"
                )
                lines.append(
                    _format_bb_level_detail(d, v, sem, max_preview=max(0, args.bb_preview))
                )
                if n_st:
                    lines.append(f"- 结构备注: {'; '.join(n_st)}\n")
                if src_root:
                    lines.append(f"- 语义检查源码根: `{src_root}`\n")
                if sem and not sem.get("skip"):
                    lines.append(
                        f"- 语义: weak_map_cols={sem.get('weak_map_cols')}, n_bb_checked={sem.get('n_bb_checked')}, "
                        f"viol_peak={sem.get('n_viol_peak')}, viol_highw={sem.get('n_viol_highw')}\n"
                    )
                elif sem and sem.get("skip"):
                    lines.append(f"- 语义: 跳过 — {sem.get('reason')}\n")
                else:
                    lines.append("- 语义: 未运行（无 check_alignment_semantics）\n")
                if n_se:
                    lines.append(f"- 语义备注: {'; '.join(n_se)}\n")
                lines.append(f"- **训练适宜性**: {suit}\n")
                lines.append("\n</details>\n")

    lines.append("\n## 运行统计\n")
    lines.append(
        f"- 预期数据集个数: {n_ds}；缺失文件: {n_missing}；无满足阈值的抽取: {n_empty_pick}；"
        f"写出明细行数: {n_rows}\n"
    )

    lines.append("\n## 结论（如何使用本报告）\n")
    lines.append(
        "1. **对齐粒度是基本块（BB）**：函数行是抽样单位；`coverage`/语义冲突/详情中的 **BB 索引与前缀预览** 才是对齐是否可疑的直接位置。\n"
    )
    lines.append(
        "2. **结构 OK + 语义 OK**：在自动化意义上与 DWARF/tree-sitter 一致，可作为强监督候选；仍建议每项目人工抽 1–2 条对照汇编。\n"
    )
    lines.append(
        "3. **语义 WARN**：矩阵主峰或高 W 与「语句行/DWARF」启发式不一致，可能是列映射弱、多文件路径或优化导致；训练时建议降权或过滤。\n"
    )
    lines.append(
        "4. **语义 SKIP**：未配置 `--src` 或源码布局与 `check_alignment_semantics` 的「按文件名扁平查找」不一致（如 OpenSSL 子目录）；可只对结构结论参考，或修正源码根/脚本后重跑。\n"
    )
    lines.append(
        "5. **下游验证**：最终仍以 **任务 loss 与验证集表现** 为准；本报告用于减少「坏标签」比例。\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(f"已写入: {out_path}")
    print(
        f"项目数={len(projects)}, 档数={len(opt_levels)}, 数据集={n_ds}, "
        f"每数据集 {k} 条, 明细行数={n_rows}, 缺失={n_missing}, 空抽取={n_empty_pick}"
    )
    print(f"语义源码映射: {list(src_map.keys())}")


if __name__ == "__main__":
    main()
