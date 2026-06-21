#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare stored GT pattern labels against current Stage2 A/B predictions.

This consumes the 100-example manual export TXT to recover the exact
``input_line`` values, loads those JSONL records, reruns current Stage2 Module A
for pattern logits, and compares ``argmax(pattern_logits)`` with the stored
JSONL ``pattern_labels`` on the same BBs.

It is intentionally scoped to the already-audited 100 examples, so it answers:
"for the bad loop/switch skeletons we saw, did AB predict labels different from
GT, or are the stored GT labels already suspicious?"
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Counter, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402
from augment_binskel_sft_from_stage2 import _load_stage2_model  # noqa: E402
from struclift.data.binskel_jsonl import (  # noqa: E402
    collate_binskel_stage2_batch,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift  # noqa: E402


PATTERN_NAMES = {
    0: "sequential",
    1: "if_header",
    2: "if_then_body",
    3: "if_else_body",
    4: "loop_header",
    5: "loop_body",
    6: "loop_exit",
    7: "switch_header",
    8: "switch_case_body",
    9: "function_prologue_epilogue",
}

EDGE_NAMES = {
    0: "COND",
    1: "UNCOND",
    2: "BACK",
    3: "CALL",
    4: "RET",
}


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "0.00%"


def _parse_export_examples(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n={20,}\n", txt)
    out: List[Dict[str, Any]] = []
    for block in blocks:
        h = re.search(
            r"^example = (\d+)\s+input_line = (\d+)\s+func = (.*?)(?:\s+verdict = (.*))?$",
            block,
            re.M,
        )
        if not h:
            continue
        c = re.search(
            r"^ctrl\(pred/source\)\s*=\s*(\d+)/(\d+).*?"
            r"gaps\(if,loop,switch\)=\(([-\d]+),([-\d]+),([-\d]+)\)",
            block,
            re.M,
        )
        verdict = (h.group(4) or "").strip()
        out.append(
            {
                "example": int(h.group(1)),
                "input_line": int(h.group(2)),
                "func": h.group(3).strip(),
                "verdict": verdict,
                "pred_ctrl": int(c.group(1)) if c else None,
                "src_ctrl": int(c.group(2)) if c else None,
                "if_gap": int(c.group(3)) if c else None,
                "loop_gap": int(c.group(4)) if c else None,
                "switch_gap": int(c.group(5)) if c else None,
                "block": block,
            }
        )
    return out


def _read_jsonl_lines(path: Path, wanted: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    want = set(int(x) for x in wanted)
    out: Dict[int, Dict[str, Any]] = {}
    if not want:
        return out
    max_line = max(want)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > max_line and len(out) >= len(want):
                break
            if i not in want:
                continue
            line = line.strip()
            if line:
                out[i] = json.loads(line)
    return out


def _n_bb(row: Dict[str, Any]) -> int:
    return effect._n_bb(row)


def _labels(row: Dict[str, Any]) -> List[int]:
    raw = row.get("pattern_labels")
    if not isinstance(raw, list):
        return []
    return [int(x) for x in raw]


def _edge_list(record: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    raw = record.get("bin_edge_index")
    typ = record.get("bin_edge_type") or []
    out: List[Tuple[int, int, int]] = []
    if isinstance(raw, list) and len(raw) == 2:
        srcs, dsts = raw
        for k, (s, d) in enumerate(zip(srcs or [], dsts or [])):
            try:
                et = int(typ[k]) if k < len(typ) else -1
                out.append((int(s), int(d), et))
            except Exception:
                continue
    return out


def _block_lines(record: Dict[str, Any], bb: int) -> List[int]:
    raw = record.get("block_src_lines") or []
    if not isinstance(raw, list) or bb < 0 or bb >= len(raw):
        return []
    vals = raw[bb]
    if not isinstance(vals, list):
        return []
    out: List[int] = []
    for x in vals:
        try:
            out.append(int(x))
        except Exception:
            pass
    return sorted(set(out))


def _line_snippets(record: Dict[str, Any], lines: Sequence[int], max_lines: int = 5) -> str:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt or not lines:
        return "-"
    parts: List[str] = []
    for ln in list(lines)[:max_lines]:
        txt = str(lt.get(int(ln), "")).strip()
        parts.append(f"L{ln}:{txt}" if txt else f"L{ln}")
    if len(lines) > max_lines:
        parts.append("...")
    return " | ".join(parts)


def _struct_loop_headers(record: Dict[str, Any]) -> Set[int]:
    feats = record.get("bin_struct_features")
    out: Set[int] = set()
    if isinstance(feats, list):
        for i, row in enumerate(feats):
            if isinstance(row, list) and len(row) > 2:
                try:
                    if float(row[2]) > 0.5:
                        out.add(i)
                except Exception:
                    pass
    return out


def _indirect_jump_bbs(record: Dict[str, Any]) -> Set[int]:
    feats = record.get("bin_struct_features")
    out: Set[int] = set()
    if isinstance(feats, list):
        for i, row in enumerate(feats):
            if isinstance(row, list) and len(row) > 9:
                try:
                    if float(row[9]) > 0.5:
                        out.add(i)
                except Exception:
                    pass
    return out


def _source_control_counts(record: Dict[str, Any]) -> Counter[str]:
    rows = effect._source_rows(record)
    active_lines = effect._active_source_lines(record)
    outline = effect._source_control_outline(
        rows,
        active_lines=active_lines,
        source_control_filter="bb-start-line",
    )
    return collections.Counter(t for t, _d, _ln in outline)


def _struct_support(record: Dict[str, Any]) -> Dict[str, Any]:
    edges = _edge_list(record)
    back_targets = {d for _s, d, et in edges if et == 2}
    struct_lh = _struct_loop_headers(record)
    outdeg: Counter[int] = collections.Counter(s for s, _d, _et in edges)
    switch_like = {bb for bb, deg in outdeg.items() if deg >= 3}
    indirect = _indirect_jump_bbs(record)
    return {
        "backedge_targets": back_targets,
        "struct_loop_headers": struct_lh,
        "expected_loop_headers": back_targets | struct_lh,
        "switch_like_outdegree_ge3": switch_like,
        "indirect_jump_bbs": indirect,
    }


@torch.inference_mode()
def _predict_labels_batch(
    model: StrucLift,
    records: List[Dict[str, Any]],
    device: torch.device,
) -> List[Tuple[List[int], List[float]]]:
    batch = collate_binskel_stage2_batch(records)
    batch = move_binskel_batch_to_device(batch, device)
    out = model.encode(
        bin_opcodes=batch["bin_opcodes"],
        bin_operand_types=batch["bin_operand_types"],
        bin_operand_values=batch["bin_operand_values"],
        bin_block_lengths=batch["bin_block_lengths"],
        bin_edge_index=batch["bin_edge_index"],
        bin_edge_type=batch["bin_edge_type"],
        bin_struct_features=batch["bin_struct_features"],
        bin_batch=batch.get("bin_batch"),
        src_token_ids=batch.get("src_token_ids"),
        src_stmt_lengths=batch.get("src_stmt_lengths"),
        src_edge_index=batch.get("src_edge_index"),
        src_edge_type=batch.get("src_edge_type"),
        src_ast_types=batch.get("src_ast_types"),
        src_compound_id=batch.get("src_compound_id"),
        src_batch=batch.get("src_batch"),
        bin_adj=batch.get("adj_bin"),
        src_adj=batch.get("adj_src_gt"),
        M_gt=None,
        run_alignment=False,
    )
    labels_all = out.pattern_logits.argmax(dim=-1).detach().cpu().tolist()
    conf_all = torch.softmax(out.pattern_logits, dim=-1).amax(dim=-1).detach().cpu().tolist()
    res: List[Tuple[List[int], List[float]]] = []
    off = 0
    for rec in records:
        n = _n_bb(rec)
        res.append((
            [int(x) for x in labels_all[off: off + n]],
            [float(x) for x in conf_all[off: off + n]],
        ))
        off += n
    return res


def _summarize_label_counts(labels: Sequence[int]) -> str:
    c = collections.Counter(int(x) for x in labels)
    return ", ".join(f"{PATTERN_NAMES.get(k, k)}={v}" for k, v in sorted(c.items())) or "-"


def _conf_summary(conf: Sequence[float]) -> str:
    vals = sorted(float(x) for x in conf)
    if not vals:
        return "n/a"
    mid = vals[len(vals) // 2]
    return f"min/median/max={vals[0]:.3f}/{mid:.3f}/{vals[-1]:.3f}"


def _per_record_result(ex: Dict[str, Any], rec: Dict[str, Any], pred: List[int], conf: List[float]) -> Dict[str, Any]:
    gt = _labels(rec)
    n = min(len(gt), len(pred))
    mism = [i for i in range(n) if gt[i] != pred[i]]
    loop_ids = {4, 5, 6}
    switch_ids = {7, 8}
    loop_mism = [i for i in mism if gt[i] in loop_ids or pred[i] in loop_ids]
    switch_mism = [i for i in mism if gt[i] in switch_ids or pred[i] in switch_ids]
    support = _struct_support(rec)
    src_counts = _source_control_counts(rec)
    gt_loop_headers = {i for i, x in enumerate(gt) if x == 4}
    pred_loop_headers = {i for i, x in enumerate(pred) if x == 4}
    expected_lh = set(support["expected_loop_headers"])
    gt_false_lh = gt_loop_headers - expected_lh
    pred_false_lh = pred_loop_headers - expected_lh
    gt_switch_headers = {i for i, x in enumerate(gt) if x == 7}
    pred_switch_headers = {i for i, x in enumerate(pred) if x == 7}
    gt_switch_cases = {i for i, x in enumerate(gt) if x == 8}
    pred_switch_cases = {i for i, x in enumerate(pred) if x == 8}
    return {
        "example": ex["example"],
        "input_line": ex["input_line"],
        "func": rec.get("func_name") or ex.get("func"),
        "source_file": rec.get("source_file"),
        "n_bb": n,
        "mismatch": mism,
        "loop_mismatch": loop_mism,
        "switch_mismatch": switch_mism,
        "match_count": n - len(mism),
        "gt": gt,
        "pred": pred,
        "conf": conf,
        "gt_counts": collections.Counter(gt),
        "pred_counts": collections.Counter(pred),
        "source_counts": src_counts,
        "backedge_targets": sorted(support["backedge_targets"]),
        "struct_loop_headers": sorted(support["struct_loop_headers"]),
        "gt_loop_headers": sorted(gt_loop_headers),
        "pred_loop_headers": sorted(pred_loop_headers),
        "gt_false_loop_headers": sorted(gt_false_lh),
        "pred_false_loop_headers": sorted(pred_false_lh),
        "gt_switch_headers": sorted(gt_switch_headers),
        "pred_switch_headers": sorted(pred_switch_headers),
        "gt_switch_cases": sorted(gt_switch_cases),
        "pred_switch_cases": sorted(pred_switch_cases),
        "switch_like_outdegree_ge3": sorted(support["switch_like_outdegree_ge3"]),
        "indirect_jump_bbs": sorted(support["indirect_jump_bbs"]),
        "export_meta": ex,
        "record": rec,
    }


def _write_detail(wf: Any, r: Dict[str, Any], max_mismatch_details: int) -> None:
    rec = r["record"]
    wf.write("\n" + "=" * 120 + "\n")
    wf.write(
        f"example={r['example']} input_line={r['input_line']} func={r['func']} n_bb={r['n_bb']}\n"
        f"source_file={r['source_file']}\n"
    )
    em = r["export_meta"]
    if em.get("pred_ctrl") is not None:
        wf.write(
            f"export_ctrl(pred/source)={em.get('pred_ctrl')}/{em.get('src_ctrl')} "
            f"gaps(if,loop,switch)=({em.get('if_gap')},{em.get('loop_gap')},{em.get('switch_gap')})\n"
        )
    wf.write(
        f"label_match={r['match_count']}/{r['n_bb']} ({_pct(r['match_count'], r['n_bb'])}) "
        f"mismatch={len(r['mismatch'])} loop_mismatch={len(r['loop_mismatch'])} "
        f"switch_mismatch={len(r['switch_mismatch'])}\n"
    )
    wf.write(f"source_counts={dict(r['source_counts'])}\n")
    wf.write(f"gt_counts={_summarize_label_counts(r['gt'])}\n")
    wf.write(f"ab_pred_counts={_summarize_label_counts(r['pred'])}\n")
    wf.write(f"ab_pred_confidence={_conf_summary(r['conf'])}\n")
    wf.write(
        f"backedge_targets={r['backedge_targets']} struct_loop_headers={r['struct_loop_headers']}\n"
        f"gt_loop_headers={r['gt_loop_headers']} pred_loop_headers={r['pred_loop_headers']}\n"
        f"gt_false_loop_headers={r['gt_false_loop_headers']} "
        f"pred_false_loop_headers={r['pred_false_loop_headers']}\n"
        f"gt_switch_headers={r['gt_switch_headers']} pred_switch_headers={r['pred_switch_headers']}\n"
        f"gt_switch_cases={r['gt_switch_cases']} pred_switch_cases={r['pred_switch_cases']}\n"
        f"switch_like_outdegree_ge3={r['switch_like_outdegree_ge3']} indirect_jump_bbs={r['indirect_jump_bbs']}\n"
    )
    wf.write("\n-- MISMATCHED BB DETAILS --\n")
    for bb in r["mismatch"][:max_mismatch_details]:
        lines = _block_lines(rec, bb)
        gt_lab = PATTERN_NAMES.get(r["gt"][bb], str(r["gt"][bb]))
        pr_lab = PATTERN_NAMES.get(r["pred"][bb], str(r["pred"][bb]))
        conf = r["conf"][bb] if bb < len(r["conf"]) else float("nan")
        wf.write(
            f"BB_{bb}: GT={gt_lab} AB={pr_lab} conf={conf:.3f} "
            f"src_lines={lines or '-'} src_text={_line_snippets(rec, lines)}\n"
        )
    if len(r["mismatch"]) > max_mismatch_details:
        wf.write(f"... {len(r['mismatch']) - max_mismatch_details} more mismatches omitted\n")

    wf.write("\n-- GT LABEL SEQUENCE --\n")
    wf.write("\n".join(
        f"BB_{i}:{PATTERN_NAMES.get(x, x)}" for i, x in enumerate(r["gt"][:260])
    ) + ("\n...\n" if len(r["gt"]) > 260 else "\n"))
    wf.write("\n-- AB PRED LABEL SEQUENCE --\n")
    wf.write("\n".join(
        f"BB_{i}:{PATTERN_NAMES.get(x, x)}" for i, x in enumerate(r["pred"][:260])
    ) + ("\n...\n" if len(r["pred"]) > 260 else "\n"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-txt", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--stage2-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-mismatch-details", type=int, default=80)
    args = ap.parse_args()

    examples = _parse_export_examples(args.export_txt.expanduser().resolve())
    records = _read_jsonl_lines(
        args.jsonl.expanduser().resolve(),
        [int(ex["input_line"]) for ex in examples],
    )
    paired = [(ex, records.get(int(ex["input_line"]))) for ex in examples]
    paired = [(ex, rec) for ex, rec in paired if rec is not None]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = _load_stage2_model(args.stage2_ckpt.expanduser().resolve(), device)
    model.eval()

    results: List[Dict[str, Any]] = []
    bs = max(1, int(args.batch_size))
    for start in range(0, len(paired), bs):
        chunk = paired[start:start + bs]
        recs = [rec for _ex, rec in chunk]
        preds = _predict_labels_batch(model, recs, device)
        for (ex, rec), (labels, conf) in zip(chunk, preds):
            results.append(_per_record_result(ex, rec, labels, conf))

    total_bb = sum(int(r["n_bb"]) for r in results)
    total_match = sum(int(r["match_count"]) for r in results)
    total_mismatch = total_bb - total_match
    loop_mismatch = sum(len(r["loop_mismatch"]) for r in results)
    switch_mismatch = sum(len(r["switch_mismatch"]) for r in results)
    gt_counts: Counter[int] = collections.Counter()
    pred_counts: Counter[int] = collections.Counter()
    confusion: Counter[Tuple[int, int]] = collections.Counter()

    gt_lh_total = pred_lh_total = gt_false_lh_total = pred_false_lh_total = 0
    gt_switch_header_total = pred_switch_header_total = 0
    gt_switch_case_total = pred_switch_case_total = 0
    funcs_gt_false_lh = funcs_pred_false_lh = 0
    funcs_gt_switch_no_case = funcs_pred_switch_no_case = 0

    for r in results:
        gt_counts.update(r["gt"])
        pred_counts.update(r["pred"])
        for g, p in zip(r["gt"], r["pred"]):
            confusion[(int(g), int(p))] += 1
        gt_lh_total += len(r["gt_loop_headers"])
        pred_lh_total += len(r["pred_loop_headers"])
        gt_false_lh_total += len(r["gt_false_loop_headers"])
        pred_false_lh_total += len(r["pred_false_loop_headers"])
        gt_switch_header_total += len(r["gt_switch_headers"])
        pred_switch_header_total += len(r["pred_switch_headers"])
        gt_switch_case_total += len(r["gt_switch_cases"])
        pred_switch_case_total += len(r["pred_switch_cases"])
        if r["gt_false_loop_headers"]:
            funcs_gt_false_lh += 1
        if r["pred_false_loop_headers"]:
            funcs_pred_false_lh += 1
        if r["gt_switch_headers"] and not r["gt_switch_cases"]:
            funcs_gt_switch_no_case += 1
        if r["pred_switch_headers"] and not r["pred_switch_cases"]:
            funcs_pred_switch_no_case += 1

    by_mismatch = sorted(results, key=lambda r: len(r["mismatch"]), reverse=True)
    by_loop = sorted(results, key=lambda r: len(r["loop_mismatch"]), reverse=True)
    by_switch = sorted(results, key=lambda r: len(r["switch_mismatch"]), reverse=True)

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== GT VS CURRENT AB LABEL COMPARISON FOR EXPORT100 =====\n")
        wf.write(f"export_txt = {args.export_txt.expanduser().resolve()}\n")
        wf.write(f"jsonl = {args.jsonl.expanduser().resolve()}\n")
        wf.write(f"stage2_ckpt = {args.stage2_ckpt.expanduser().resolve()}\n")
        wf.write(f"examples_in_export = {len(examples)}\n")
        wf.write(f"examples_compared = {len(results)}\n")
        wf.write(f"total_bbs = {total_bb}\n")
        wf.write(f"label_match = {total_match}/{total_bb} ({_pct(total_match, total_bb)})\n")
        wf.write(f"label_mismatch = {total_mismatch}/{total_bb} ({_pct(total_mismatch, total_bb)})\n")
        wf.write(f"loop_related_mismatches = {loop_mismatch}\n")
        wf.write(f"switch_related_mismatches = {switch_mismatch}\n")
        wf.write("\n===== LABEL COUNTS =====\n")
        for lab in sorted(PATTERN_NAMES):
            wf.write(
                f"{lab:2d} {PATTERN_NAMES[lab]:28s} "
                f"GT={gt_counts.get(lab, 0):6d}  AB={pred_counts.get(lab, 0):6d}\n"
            )
        wf.write("\n===== LOOP/SWITCH STRUCTURAL CHECK ON SAME 100 =====\n")
        wf.write(f"gt_loop_headers = {gt_lh_total}\n")
        wf.write(f"ab_pred_loop_headers = {pred_lh_total}\n")
        wf.write(f"gt_false_loop_headers_vs_backedge_or_struct = {gt_false_lh_total}/{gt_lh_total} ({_pct(gt_false_lh_total, gt_lh_total)})\n")
        wf.write(f"ab_pred_false_loop_headers_vs_backedge_or_struct = {pred_false_lh_total}/{pred_lh_total} ({_pct(pred_false_lh_total, pred_lh_total)})\n")
        wf.write(f"funcs_with_gt_false_loop_header = {funcs_gt_false_lh}/{len(results)}\n")
        wf.write(f"funcs_with_ab_false_loop_header = {funcs_pred_false_lh}/{len(results)}\n")
        wf.write(f"gt_switch_headers = {gt_switch_header_total}\n")
        wf.write(f"ab_pred_switch_headers = {pred_switch_header_total}\n")
        wf.write(f"gt_switch_case_body = {gt_switch_case_total}\n")
        wf.write(f"ab_pred_switch_case_body = {pred_switch_case_total}\n")
        wf.write(f"funcs_gt_switch_header_no_case_body = {funcs_gt_switch_no_case}/{len(results)}\n")
        wf.write(f"funcs_ab_switch_header_no_case_body = {funcs_pred_switch_no_case}/{len(results)}\n")

        wf.write("\n===== TOP CONFUSIONS GT -> AB =====\n")
        for (g, p), c in confusion.most_common(40):
            if g == p:
                continue
            wf.write(
                f"{PATTERN_NAMES.get(g, g):28s} -> {PATTERN_NAMES.get(p, p):28s} {c}\n"
            )

        wf.write("\n===== TOP EXAMPLES BY TOTAL LABEL MISMATCH =====\n")
        for r in by_mismatch[:25]:
            wf.write(
                f"ex={r['example']:3d} line={r['input_line']:6d} func={str(r['func'])[:32]:32s} "
                f"match={r['match_count']}/{r['n_bb']} mism={len(r['mismatch'])} "
                f"loop_mism={len(r['loop_mismatch'])} switch_mism={len(r['switch_mismatch'])} "
                f"gt_false_lh={len(r['gt_false_loop_headers'])} ab_false_lh={len(r['pred_false_loop_headers'])}\n"
            )

        wf.write("\n===== TOP EXAMPLES BY LOOP LABEL MISMATCH =====\n")
        for r in by_loop[:25]:
            wf.write(
                f"ex={r['example']:3d} line={r['input_line']:6d} func={str(r['func'])[:32]:32s} "
                f"loop_mism={len(r['loop_mismatch'])} "
                f"gt_lh={r['gt_loop_headers']} ab_lh={r['pred_loop_headers']} "
                f"backedge={r['backedge_targets']} gt_false={r['gt_false_loop_headers']} "
                f"ab_false={r['pred_false_loop_headers']}\n"
            )

        wf.write("\n===== TOP EXAMPLES BY SWITCH LABEL MISMATCH =====\n")
        for r in by_switch[:25]:
            wf.write(
                f"ex={r['example']:3d} line={r['input_line']:6d} func={str(r['func'])[:32]:32s} "
                f"switch_mism={len(r['switch_mismatch'])} "
                f"gt_sw={r['gt_switch_headers']} ab_sw={r['pred_switch_headers']} "
                f"gt_case={r['gt_switch_cases']} ab_case={r['pred_switch_cases']}\n"
            )

        wf.write("\n===== DETAILED EXAMPLES =====\n")
        for r in by_mismatch[: int(min(len(by_mismatch), 40))]:
            _write_detail(wf, r, int(args.max_mismatch_details))

    print(f"saved_to = {out}")
    print(f"examples_compared = {len(results)}")
    print(f"label_match = {total_match}/{total_bb} ({_pct(total_match, total_bb)})")
    print(f"loop_related_mismatches = {loop_mismatch}")
    print(f"switch_related_mismatches = {switch_mismatch}")
    print(f"gt_false_loop_headers = {gt_false_lh_total}/{gt_lh_total}")
    print(f"ab_pred_false_loop_headers = {pred_false_lh_total}/{pred_lh_total}")
    print(f"gt_switch_headers/cases = {gt_switch_header_total}/{gt_switch_case_total}")
    print(f"ab_pred_switch_headers/cases = {pred_switch_header_total}/{pred_switch_case_total}")


if __name__ == "__main__":
    main()
