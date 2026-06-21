#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3 复原效果评测（jsonl → A/B/C + LLM → 填槽 C 代码）。

**默认 ``--profile deploy``（推理）**：仅二进制 jsonl → Module A 骨架 → Stage3 **逐槽自回归**
填 C 文本；**不读源码、不用 M_gt**。产物是 ``recovered.c``（控制流骨架 + 槽内语句片段）。

``--profile train``：仅调试/算分（可选 ``--metrics-from-gt`` 写 reference，不参与前向）。

无源码推理::

  scripts/build_binskel_infer_jsonl.py
  scripts/infer_binskel_pipeline.py   # 等价 deploy

与训练一致的完整链路（仓库实现）::

  ELF + 源码
    → build_binskel_dataset_md（CFG、指令特征、规则 ``pattern_labels``、对齐矩阵）
    → augment_binskel_sft（CST 骨架 + FILL 序列 + ``sft_*``）
    → Stage3 训练：forward 里 Module A 算 ``pattern_logits``；``run_alignment=True`` 时
      Module B 用 **A 的 argmax 模式**（非 jsonl 里的 ``pattern_labels``）做对齐；
      prompt 里的骨架却来自增广时的 **jsonl GT pattern_labels**（存在 train/serve 差异）。
    → 本脚本评测：加载 Stage2/3 ckpt，对 jsonl 一条样本做 encode + 一次前向 + 贪心填槽。

本脚本**会做的**::

  - 加载 StrucLift（A/B/C adapter）+ HF 基座；``graph_kv`` 来自 **Module A 前向**（``bin_only``）
    或 A+B+``M_gt``（``gt_src``，上界）。
  - ``bin_skeleton``：prompt 不含槽位 GT token；``labels`` 仍用 GT 仅作对比指标。
  - 输出 ``pred.c`` / ``gt.c`` / ``compile_reward``。

本脚本**不会做的**（端到端反编译缺口）::

  - 不从 ELF 现场 disassemble / 建 CFG（请先用 ``build_binskel_dataset_md.py``）。
  - 默认骨架用 jsonl 的 **GT ``pattern_labels``**（数据集规则+对齐细化），不是 Module A 预测；
    用 ``--skeleton-patterns module_a`` 改为 A 的 argmax（更接近部署，但与 Stage3 训练 prompt 不一致）。
  - 非自回归多步 decode（仅单次 forward + 监督位 argmax）；无 GT 时需
    ``--infer-slot-tokens`` 在每个 FILL 后预留 pad，否则**不会**产生槽位文本。
  - ``recovered.c`` 会去掉 ``<|STRUCLIFT_FILL_*|>``，只保留槽内预测 + 骨架字面量。
  - ``bin_only`` 时 prompt 里的 ``<|binary_context|>`` 仍是 opcode 摘要，**不含**源码对齐峰
    （``enrich_skeleton_slots_with_alignment_matrix`` 未在增广路径调用）。

评测模式（``--eval-mode``）::

  sft_greedy     使用 jsonl 里已 augment 的 ``sft_input_ids``（与训练 batch 最一致）。

  bin_skeleton   现场重建 prompt；FILL 槽不含 GT token。

骨架模式（``--skeleton-patterns``，仅 ``bin_skeleton``）::

  jsonl_gt       与 ``augment_binskel_sft`` 相同：jsonl 的 ``pattern_labels``。
  module_a       先 ``encode`` 取 ``pattern_logits.argmax`` 再建树（部署向）。

对齐（``--alignment``）::

  gt_src         Module B + jsonl 源码图 + ``M_gt``（Stage3 训练同款，上界）。
  bin_only       跳过 B；``graph_kv`` 仅二进制节点嵌入。

示例（卡 7、10 条 val、当前 best_stage3）::

  cd /data/chaoni/WQM/model_code/structlift
  CUDA_VISIBLE_DEVICES=7 python3 scripts/eval_stage3_binskel.py \\
    --init-from /data/chaoni/WQM/checkpoints/stage3_o0_deepseek67_p20_2gpu_b4/best_stage3.pt \\
    --jsonl /data/chaoni/WQM/datasets/C_val_o0_deepseek_coder6.7b_p20.jsonl \\
    --llm /data/chaoni/WQM/models/deepseek-coder-6.7b-instruct \\
    --tokenizer-saved /data/chaoni/WQM/datasets/structlift_tokenizer_deepseek_6.7b_instruct_fill \\
    --max-samples 10 \\
    --eval-mode bin_skeleton \\
    --alignment bin_only \\
    --out-dir /tmp/stage3_eval_10funcs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    collate_binskel_batch,
    collate_binskel_stage3_batch,
    move_binskel_batch_to_device,
)
from struclift.models.module_c import SlotWeightBuilder, StructureConditionedDecoder
from struclift.models.module_d import compile_reward
from struclift.models.struclift import StrucLift, batched_graph_kv_from_nodes
from struclift.utils.state_dict_compat import load_state_dict_skip_shape_mismatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_FILL_RE = re.compile(r"<FILL_[A-Z_]+:slot_\d+>")


def _load_autoconfig():
    path = SCRIPT_DIR / "train_stage3_binskel_autoconfig.py"
    spec = importlib.util.spec_from_file_location("_t3_autoconfig_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_llm_pretrained_source(llm_raw: str) -> str:
    """本地 HF 目录 → 绝对路径；否则当作 Hub id。"""
    raw = llm_raw.strip()
    if not raw:
        raise SystemExit("--llm 为空")
    cand = Path(raw).expanduser()
    try:
        resolved = cand.resolve(strict=False)
    except OSError:
        resolved = cand
    if resolved.is_dir() and (resolved / "config.json").is_file():
        return str(resolved)
    if resolved.is_absolute() or raw.startswith(("./", "../", "~/")):
        logger.error("本地 --llm 无效或缺少 config.json: %s", resolved)
        raise SystemExit(1)
    return raw


def _load_augment():
    path = SCRIPT_DIR / "augment_binskel_sft.py"
    spec = importlib.util.spec_from_file_location("_augment_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_gt_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """按 ``func_name`` 索引带源码字段的样本（用于无源码推理 jsonl 的评测对比）。"""
    idx: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = rec.get("func_name")
            if isinstance(fn, str) and fn:
                idx[fn] = rec
    return idx


def _merge_gt_fields(rec: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    """把监督/对比所需字段并入推理样本（不改 ``bin_*``）。"""
    out = dict(rec)
    for k in (
        "source_file",
        "block_src_lines",
        "block_src_confidence",
        "alignment_matrix",
        "alignment_confidence",
        "src_token_ids",
        "src_stmt_lengths",
        "src_edge_index",
        "src_edge_type",
        "src_ast_types",
        "sft_input_ids",
        "sft_labels",
        "slot_types",
        "sft_fill_positions",
        "sft_slot_bb_indices",
    ):
        if k in gt:
            out[k] = gt[k]
    return out


def _apply_source_rewrites_record(
    rec: Dict[str, Any],
    prefix_pairs: Optional[List[Tuple[str, str]]],
    aug_mod: Any,
) -> Dict[str, Any]:
    if not prefix_pairs:
        return rec
    out = dict(rec)
    sf = out.get("source_file")
    if isinstance(sf, str) and sf.strip():
        out["source_file"] = aug_mod._rewrite_source_path_string(sf, prefix_pairs)
    return out


def _find_subseq(hay: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(hay):
        return -1
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    return -1


def _skeleton_head_body_len(tokenizer: Any, input_ids: List[int]) -> Tuple[int, int]:
    """从 ``sft_input_ids`` 定位 ``<|skeleton|>`` body 段（不含 binary_context）。"""
    for sk_open, sk_close in (
        ("<|skeleton|>\n", "\n<|/skeleton|>"),
        ("<|skeleton|>", "<|/skeleton|>"),
    ):
        open_ids = tokenizer.encode(sk_open, add_special_tokens=False)
        close_ids = tokenizer.encode(sk_close, add_special_tokens=False)
        i = _find_subseq(input_ids, open_ids)
        if i < 0:
            continue
        head_len = i + len(open_ids)
        tail = input_ids[head_len:]
        j = _find_subseq(tail, close_ids)
        if j >= 0:
            return head_len, j
        out_ids = tokenizer.encode("<|output|>", add_special_tokens=False)
        k = _find_subseq(tail, out_ids)
        return head_len, (k if k >= 0 else len(tail))
    return 0, 0


def _export_recovered_skeleton_c(
    tokenizer: Any,
    token_ids: List[int],
    fill_ids: List[int],
    *,
    head_len: Optional[int] = None,
    body_len: Optional[int] = None,
) -> str:
    """只导出 skeleton 段 C（先字符串切边界，再 token 级去 FILL）。"""
    if head_len is None or body_len is None:
        hl, bl = _skeleton_head_body_len(tokenizer, token_ids)
    else:
        hl, bl = int(head_len), int(body_len)
    if bl > 0:
        rendered = _render_filled_skeleton_body(
            tokenizer, token_ids, head_len=hl, body_len=bl, fill_ids=fill_ids,
        )
        if _clean_skeleton_text(rendered).strip():
            return _clean_skeleton_text(rendered)
        raw_body = _decode_skeleton_body(
            tokenizer, token_ids, head_len=hl, body_len=bl,
        )
        if _clean_skeleton_text(raw_body).strip():
            return _clean_skeleton_text(raw_body)
    return ""


def _source_readable(
    rec: Dict[str, Any],
    source_root: Optional[Path],
    aug_mod: Any,
) -> bool:
    p = aug_mod._resolve_source_path(rec, source_root)
    return p is not None and p.is_file()


def _resolve_eval_profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.profile == "train":
        return {
            "alignment": "gt_src",
            "skeleton_patterns": "jsonl_gt",
            "prefer_sft_greedy": True,
        }
    if args.profile == "deploy":
        return {
            "alignment": "bin_only",
            "skeleton_patterns": "module_a",
            "prefer_sft_greedy": False,
        }
    return {
        "alignment": args.alignment,
        "skeleton_patterns": args.skeleton_patterns,
        "prefer_sft_greedy": args.eval_mode in ("auto", "sft_greedy"),
    }


def _pick_eval_mode(
    args: argparse.Namespace,
    rec: Dict[str, Any],
    prof: Dict[str, Any],
) -> str:
    if args.profile == "deploy":
        return "bin_skeleton"
    if args.eval_mode != "auto":
        return str(args.eval_mode)
    if prof.get("prefer_sft_greedy") and rec.get("sft_input_ids") and rec.get("sft_labels"):
        return "sft_greedy"
    return "bin_skeleton"


def _ensure_train_sft_fields(
    rec: Dict[str, Any],
    aug_mod: Any,
    tokenizer: Any,
    fill_ids: List[int],
    source_root: Optional[Path],
    prefix_pairs: Optional[List[Tuple[str, str]]],
    max_seq_len: Optional[int],
) -> Dict[str, Any]:
    """``profile=train`` 时尽量得到与 augment 一致的 ``sft_*``（无则现场 process_record）。"""
    if rec.get("sft_input_ids") and rec.get("sft_labels"):
        return rec
    new_r, err = aug_mod.process_record(
        rec,
        tokenizer,
        fill_ids,
        source_root,
        emit_rl_fields=False,
        num_patterns=10,
        mask_prompt_labels=True,
        mask_non_fill_labels=True,
        mask_supervised_inputs=False,
        max_seq_len=max_seq_len,
        prefix_rewrites=prefix_pairs,
    )
    if new_r is None:
        raise ValueError(err or "augment process_record 失败")
    return new_r


def _read_jsonl_records(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= max_samples:
                break
    return out


def _fill_token_id_list(tokenizer: Any, max_fill: int) -> List[int]:
    needed = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(max_fill)]
    vocab = tokenizer.get_vocab()
    missing = [t for t in needed if t not in vocab]
    if missing:
        tokenizer.add_tokens(missing)
    return [int(tokenizer.convert_tokens_to_ids(t)) for t in needed]


def _pad_id(tokenizer: Any) -> int:
    p = tokenizer.pad_token_id
    if p is None:
        p = tokenizer.eos_token_id
    return int(p or 0)


@torch.inference_mode()
def _pattern_labels_from_module_a(
    model: StrucLift,
    record: Dict[str, Any],
    device: torch.device,
) -> List[int]:
    """单样本：仅跑 Module A，返回离散模式标签（部署推理仅需 ``bin_*``）。"""
    batch = move_binskel_batch_to_device(
        collate_binskel_batch([record]), device,
    )
    enc = model.encode(
        bin_opcodes=batch["bin_opcodes"],
        bin_operand_types=batch["bin_operand_types"],
        bin_operand_values=batch["bin_operand_values"],
        bin_block_lengths=batch["bin_block_lengths"],
        bin_edge_index=batch["bin_edge_index"],
        bin_edge_type=batch["bin_edge_type"],
        bin_struct_features=batch["bin_struct_features"],
        bin_batch=batch.get("bin_batch"),
        run_alignment=False,
    )
    return [int(x) for x in enc.pattern_logits.argmax(dim=-1).tolist()]


def _interleave_skeleton_for_eval(
    skeleton_text: str,
    slots: List[Any],
    gt_texts: List[str],
    fill_ids: List[int],
    tokenizer: Any,
    *,
    infer_only_body: bool,
    infer_slot_tokens: int,
    pad_id: int,
) -> Tuple[List[int], List[bool], List[int]]:
    """与 augment 一致，但纯推理时在每个 FILL 后插入 ``infer_slot_tokens`` 个 pad 供贪心填槽。"""
    ids: List[int] = []
    supervised: List[bool] = []
    fill_pos_in_body: List[int] = []
    pos = 0
    si = 0
    for m in _FILL_RE.finditer(skeleton_text):
        chunk_before = skeleton_text[pos : m.start()]
        if chunk_before:
            tok = tokenizer.encode(chunk_before, add_special_tokens=False)
            ids.extend(tok)
            supervised.extend([False] * len(tok))
        if si >= len(slots):
            raise ValueError("FILL 占位符数量与 slots 不一致")
        fid = fill_ids[si] if si < len(fill_ids) else fill_ids[-1]
        fill_pos_in_body.append(len(ids))
        ids.append(int(fid))
        supervised.append(False)
        gt = gt_texts[si]
        if gt:
            gt_tok = tokenizer.encode(gt, add_special_tokens=False)
            ids.extend(gt_tok)
            supervised.extend([True] * len(gt_tok))
        elif infer_only_body and infer_slot_tokens > 0:
            ids.extend([int(pad_id)] * int(infer_slot_tokens))
            supervised.extend([True] * int(infer_slot_tokens))
        si += 1
        pos = m.end()
    tail = skeleton_text[pos:]
    if tail:
        tok = tokenizer.encode(tail, add_special_tokens=False)
        ids.extend(tok)
        supervised.extend([False] * len(tok))
    if si != len(slots):
        raise ValueError(f"slots={len(slots)} 与占位符匹配数={si} 不一致")
    return ids, supervised, fill_pos_in_body


def _fill_id_set(fill_ids: List[int]) -> set:
    return {int(x) for x in fill_ids}


def _fill_positions_in_ids(
    input_ids: List[int],
    fill_ids: List[int],
    tokenizer: Any,
) -> List[int]:
    fid = _fill_id_set(fill_ids)
    return [
        i
        for i, t in enumerate(input_ids)
        if _token_is_fill_marker(int(t), fid, tokenizer)
    ]


def _sync_slots_to_fills(
    input_ids: List[int],
    slot_types: List[str],
    fill_positions: List[int],
    bb_indices: List[List[int]],
    fill_ids: List[int],
    tokenizer: Any,
) -> Tuple[List[str], List[int], List[List[int]]]:
    """截断/分词后 FILL 个数可能少于 ``skel.slots``；与 token 序列对齐。"""
    actual = _fill_positions_in_ids(input_ids, fill_ids, tokenizer)
    n = len(actual)
    st = [str(x) for x in slot_types[:n]]
    while len(st) < n:
        st.append("FILL_STMT")
    bb = [list(x) for x in bb_indices[:n]]
    while len(bb) < n:
        bb.append([])
    return st, actual, bb


def _token_is_fill_marker(
    tid: int,
    fill_ids: set,
    tokenizer: Any,
) -> bool:
    """FILL 占位符 id（含未进 fill_ids 表但 decode 带 STRUCLIFT_FILL 的 token）。"""
    if int(tid) in fill_ids:
        return True
    try:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
    except Exception:
        return False
    u = piece.upper()
    return "STRUCLIFT_FILL" in u or "STRUCTLIFT_FILL" in u


def _render_filled_skeleton_body(
    tokenizer: Any,
    token_ids: List[int],
    *,
    head_len: int,
    body_len: int,
    fill_ids: List[int],
) -> str:
    """把 body 里的 FILL 占位符换成其后模型预测 token 的可读 C（不保留 ``<|STRUCLIFT_FILL_*|>``）。"""
    end = min(len(token_ids), head_len + body_len)
    if head_len >= end:
        return ""
    body = token_ids[head_len:end]
    fid = _fill_id_set(fill_ids)
    parts: List[str] = []
    i = 0
    while i < len(body):
        tid = int(body[i])
        if _token_is_fill_marker(tid, fid, tokenizer):
            i += 1
            slot_toks: List[int] = []
            while i < len(body) and not _token_is_fill_marker(int(body[i]), fid, tokenizer):
                slot_toks.append(int(body[i]))
                i += 1
            if slot_toks:
                parts.append(
                    tokenizer.decode(slot_toks, skip_special_tokens=True),
                )
            continue
        chunk: List[int] = []
        while i < len(body) and not _token_is_fill_marker(int(body[i]), fid, tokenizer):
            chunk.append(int(body[i]))
            i += 1
        if chunk:
            parts.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return "".join(parts)


def _render_prediction_on_masked_template(
    tokenizer: Any,
    input_ids: List[int],
    pred_ids: List[int],
    labels: List[int],
    fill_ids: List[int],
    *,
    head_len: int,
    body_len: int,
) -> str:
    """Render C from the masked SFT template using predictions at label sites.

    This is the most faithful visualization for ``sft_greedy``: fill_acc is
    computed exactly at ``labels != -100`` positions, so recovered.c should be
    reconstructed from the same positions instead of relying on decoded prompt
    boundary heuristics.
    """
    merged = [int(x) for x in input_ids]
    n = min(len(merged), len(pred_ids), len(labels))
    for i in range(n):
        if int(labels[i]) != -100:
            merged[i] = int(pred_ids[i])

    hl = int(head_len)
    bl = int(body_len)
    if hl <= 0 or bl <= 0 or hl >= len(merged):
        hl, bl = _skeleton_head_body_len(tokenizer, merged)

    body_text = ""
    if hl > 0 and bl > 0 and hl < len(merged):
        end = min(len(merged), hl + bl)
        body_text = tokenizer.decode(merged[hl:end], skip_special_tokens=False)
    if not body_text.strip():
        body_text = _extract_skeleton_c_text(
            tokenizer.decode(merged, skip_special_tokens=False),
        )
    return _clean_skeleton_text(body_text)


def _render_gt_from_masked_template(
    tokenizer: Any,
    input_ids: List[int],
    labels: List[int],
    fill_ids: List[int],
    *,
    head_len: int,
    body_len: int,
) -> str:
    """Render the supervised SFT target directly from existing input/labels."""
    merged = [int(x) for x in input_ids]
    n = min(len(merged), len(labels))
    for i in range(n):
        if int(labels[i]) != -100:
            merged[i] = int(labels[i])
    hl = int(head_len)
    bl = int(body_len)
    if hl <= 0 or bl <= 0 or hl >= len(merged):
        hl, bl = _skeleton_head_body_len(tokenizer, merged)
    if hl > 0 and bl > 0 and hl < len(merged):
        end = min(len(merged), hl + bl)
        return _clean_skeleton_text(
            tokenizer.decode(merged[hl:end], skip_special_tokens=False),
        )
    return _clean_skeleton_text(
        _extract_skeleton_c_text(tokenizer.decode(merged, skip_special_tokens=False)),
    )


def _clean_skeleton_text(text: str) -> str:
    """Remove prompt/special-token remnants from exported skeleton C."""
    if not text:
        return ""
    for tag in (
        "<|system|>",
        "<|/system|>",
        "<|binary_info|>",
        "<|/binary_info|>",
        "<|binary_context|>",
        "<|/binary_context|>",
        "<|skeleton|>",
        "<|/skeleton|>",
        "<|output|>",
    ):
        text = text.replace(tag, "")
    text = re.sub(r"<\|STRUCLIFT_FILL_\d{4}\|>", "", text)
    text = text.replace("<|STRUCLIFT_MASK|>", "")
    # Special-token boundaries can eat whitespace during decode, e.g.
    # ``<FILL_TYPE> func`` rendered as ``intfunc``.  Keep these fixes narrow:
    # they improve readability without hiding genuinely bad model predictions.
    ret_words = (
        "void|char|short|int|long|float|double|bool|_Bool|size_t|ssize_t|"
        "idx_t|ptrdiff_t|intptr_t|uintptr_t|int8_t|int16_t|int32_t|int64_t|"
        "uint8_t|uint16_t|uint32_t|uint64_t|uintmax_t|intmax_t|FILE"
    )
    spec_words = "static|extern|inline|const|volatile|signed|unsigned|restrict"
    text = re.sub(rf"\b({ret_words})(?=const\b)", r"\1 ", text)
    text = re.sub(rf"\b({spec_words})(?=({ret_words})\b)", r"\1 ", text)
    text = re.sub(r"\blong(?=long\b)", "long ", text)
    text = re.sub(r"\bstruct(?=[A-Za-z_][A-Za-z0-9_]*\b)", "struct ", text)
    text = re.sub(r"\benum(?=[A-Za-z_][A-Za-z0-9_]*\b)", "enum ", text)
    text = re.sub(r"\bunion(?=[A-Za-z_][A-Za-z0-9_]*\b)", "union ", text)
    text = re.sub(rf"\b({ret_words})\s*(\*+)\s*([A-Za-z_])", r"\1 \2\3", text)
    text = re.sub(rf"\b({ret_words})(?=func\s*\()", r"\1 ", text)
    text = re.sub(r"(\*+)(?=func\s*\()", r"\1 ", text)
    text = re.sub(
        r"\breturn(?=(?:0x[0-9A-Fa-f]+|\d|true\b|false\b|NULL\b|nullptr\b|sizeof\b|[(!*&~+\-]|[A-Za-z_]))",
        "return ",
        text,
    )
    text = re.sub(r"\b(if|while|for|switch)(?=\()", r"\1 ", text)
    text = re.sub(r"\bint(argc|fd|rc|n|i|j|k)\b", r"int \1", text)
    text = re.sub(r"\bsize_t([A-Za-z_][A-Za-z0-9_]*)\b", r"size_t \1", text)
    text = re.sub(r"\bidx_t([A-Za-z_][A-Za-z0-9_]*)\b", r"idx_t \1", text)
    text = re.sub(r"\bssize_t([A-Za-z_][A-Za-z0-9_]*)\b", r"ssize_t \1", text)
    text = re.sub(r"\bptrdiff_t([A-Za-z_][A-Za-z0-9_]*)\b", r"ptrdiff_t \1", text)
    text = re.sub(r"\bchar(\*+)(argv|envp|name|path|buf|str)\b", r"char \1\2", text)
    text = re.sub(r"\bchar\s+const\s*(\*+)\s*([A-Za-z_])", r"char const \1\2", text)
    text = re.sub(r"\bvoid\s+const\s*(\*+)\s*([A-Za-z_])", r"void const \1\2", text)
    text = re.sub(r"\bconst char(\*+)(name|path|buf|str)\b", r"const char \1\2", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\)\s*\{", ") {", text)
    text = re.sub(r"\{\s*\{", "{\n    {", text)
    text = re.sub(r";\s*//\s*alignedto", ";  // aligned to ", text)
    text = re.sub(r"//\s*alignedto", "// aligned to ", text)
    text = re.sub(r"BB_(\d+)\(", r"BB_\1 (", text)
    return text.strip()


def _build_infer_input_ids(
    record: Dict[str, Any],
    tokenizer: Any,
    fill_ids: List[int],
    *,
    infer_only_body: bool,
    infer_slot_tokens: int,
    pad_id: int,
    max_seq_len: Optional[int],
    aug_mod: Any,
    pattern_labels_override: Optional[List[int]] = None,
    source_root: Optional[Path] = None,
    prefix_rewrites: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[int], List[int], List[str], List[int], List[List[int]], int, int]:
    """构造 ``input_ids`` / ``labels``；并返回 ``(head_len, body_len)`` 供只解码骨架 body。"""
    record = _apply_source_rewrites_record(record, prefix_rewrites, aug_mod)
    pl = pattern_labels_override if pattern_labels_override is not None else record["pattern_labels"]
    bei = record["bin_edge_index"]
    n_bb = len(pl)
    hints = aug_mod._bb_opcode_hints(record)
    pl_t = torch.tensor([int(x) for x in pl], dtype=torch.int64)
    edge_index = torch.tensor(
        [[int(x) for x in bei[0]], [int(x) for x in bei[1]]],
        dtype=torch.int64,
    )
    skel = aug_mod.build_skeleton_for_binary_cfg(pl_t, edge_index, n_bb, hints)

    src_path = aug_mod._resolve_source_path(record, source_root)
    block_src_lines = record.get("block_src_lines") or []
    lt: Dict[int, str] = aug_mod._line_to_text(src_path) if src_path else {}
    if lt and hasattr(aug_mod, "_strip_block_comments_from_lt"):
        lt = aug_mod._strip_block_comments_from_lt(lt)
    teacher_signature = aug_mod._extract_source_signature(record, lt)
    prompt_signature = (
        teacher_signature
        if aug_mod._signature_is_confident(teacher_signature)
        else (aug_mod.UNKNOWN_RET, aug_mod.UNKNOWN_PARAMS, teacher_signature[2])
    )
    gt_texts = (
        [""] * len(skel.slots)
        if infer_only_body
        else [aug_mod._gt_for_slot(s, block_src_lines, lt, teacher_signature) for s in skel.slots]
    )
    slot_types = [s.slot_type for s in skel.slots]

    system = (
        "<|system|>\n"
        "You are a decompiler. Fill in the placeholders in the code skeleton\n"
        "based on the binary context provided for each slot.\n"
        "<|/system|>\n"
    )
    bin_info = aug_mod._binary_info_block(record, prompt_signature)
    ctx_lines = []
    if hasattr(aug_mod, "_readable_asm_block"):
        asm_block, _ = aug_mod._readable_asm_block(record)
        if asm_block:
            ctx_lines.append(asm_block)
    ctx_lines.extend(aug_mod._slot_binary_context_line(s) for s in skel.slots)
    ctx_block = "<|binary_context|>\n" + "\n".join(ctx_lines) + "\n<|/binary_context|>"
    skeleton_text = (
        aug_mod._format_skeleton_text_for_prompt(skel.skeleton_text)
        if hasattr(aug_mod, "_format_skeleton_text_for_prompt")
        else skel.skeleton_text
    )

    eff_max = aug_mod._effective_sft_max_len(tokenizer, max_seq_len)
    old_mx = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = 10**9
        head_text = system.strip() + "\n\n" + bin_info.strip() + "\n\n<|skeleton|>\n"
        tail_text = "\n<|/skeleton|>\n\n" + ctx_block + "\n\n<|output|>\n"
        head_ids = tokenizer.encode(head_text, add_special_tokens=False)
        body_ids, body_sup, fill_pos_body = _interleave_skeleton_for_eval(
            skeleton_text,
            skel.slots,
            gt_texts,
            fill_ids,
            tokenizer,
            infer_only_body=infer_only_body,
            infer_slot_tokens=infer_slot_tokens,
            pad_id=pad_id,
        )
        tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)
    finally:
        if old_mx is not None:
            tokenizer.model_max_length = old_mx

    input_ids = head_ids + body_ids + tail_ids
    if len(input_ids) > eff_max:
        # binary_context 已在 head 里；旧格式这里曾优先缩短 tail context。
        tail_min = tokenizer.encode("\n<|/skeleton|>\n\n<|output|>\n", add_special_tokens=False)
        fixed = len(head_ids) + len(body_ids) + len(tail_min)
        if fixed <= eff_max:
            input_ids = head_ids + body_ids + tail_min
        else:
            body_budget = max(0, eff_max - len(head_ids) - len(tail_min))
            body_ids = body_ids[:body_budget]
            input_ids = head_ids + body_ids + tail_min
        if len(input_ids) > eff_max:
            input_ids = input_ids[:eff_max]

    labels = list(input_ids)
    supervised_mask = [False] * len(head_ids) + body_sup + [False] * len(tail_ids)
    for i, sup in enumerate(supervised_mask):
        if not sup:
            labels[i] = -100

    fill_positions = [len(head_ids) + int(p) for p in fill_pos_body]
    bb_indices = [[int(x) for x in s.aligned_bb_indices] for s in skel.slots]
    return (
        input_ids,
        labels,
        slot_types,
        fill_positions,
        bb_indices,
        len(head_ids),
        len(body_ids),
    )


def _decode_skeleton_body(
    tokenizer: Any,
    token_ids: List[int],
    *,
    head_len: int,
    body_len: int,
) -> str:
    """只解码 ``<|skeleton|>`` 与 ``<|/skeleton|>`` 之间的 body token（避免整段 prompt 写入 .c）。"""
    end = min(len(token_ids), head_len + body_len)
    if head_len >= end:
        return ""
    return tokenizer.decode(token_ids[head_len:end], skip_special_tokens=False)


def _mask_gt_tokens_in_input(
    input_ids: List[int],
    labels: List[int],
    *,
    mask_id: int,
) -> List[int]:
    """``bin_skeleton``：序列长度与带 GT 的 labels 一致，但 input 里槽位真值改为 pad（不泄露 GT token）。"""
    out = list(input_ids)
    for i, lb in enumerate(labels):
        if i < len(out) and lb != -100:
            out[i] = int(mask_id)
    return out


def _greedy_fill_from_logits(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """在 ``labels != -100`` 的 next-token 位置用 argmax 预测，其余位置保留 input。"""
    B, L = input_ids.shape
    pred = input_ids.clone()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_pred = pred[:, 1:].clone()
    mask = shift_labels != -100
    if mask.any():
        shift_pred[mask] = shift_logits[mask].argmax(dim=-1)
    pred[:, 1:] = shift_pred
    return pred


def _extract_skeleton_c_text(full_text: str) -> str:
    """提案式 prompt 里，可读的 C 在 ``<|skeleton|>`` … ``<|/skeleton|>`` 之间（非 ``<|output|>`` 后）。"""
    for sk_open, sk_close in (
        ("<|skeleton|>\n", "\n<|/skeleton|>"),
        ("<|skeleton|>", "<|/skeleton|>"),
    ):
        i = full_text.find(sk_open)
        if i < 0:
            continue
        start = i + len(sk_open)
        j = full_text.find(sk_close, start)
        if j < 0:
            j = full_text.find("<|output|>", start)
        return full_text[start:j].strip() if j >= 0 else full_text[start:].strip()
    return ""


def _extract_output_text(full_text: str) -> str:
    """兼容旧调用：实际返回骨架段 C 文本。"""
    return _extract_skeleton_c_text(full_text)


def _fill_token_accuracy(
    pred_ids: torch.Tensor,
    labels: torch.Tensor,
    fill_token_ids: torch.Tensor,
) -> float:
    """监督位置上 token 准确率（仅统计 labels!=-100）。"""
    valid = labels[:, 1:] != -100
    if not valid.any():
        return 0.0
    pred_tok = pred_ids[:, 1:][valid]
    gold_tok = labels[:, 1:][valid]
    return float((pred_tok == gold_tok).float().mean().item())


def _slot_prediction_debug(
    tokenizer: Any,
    input_ids: List[int],
    pred_ids: List[int],
    labels: List[int],
    fill_ids: List[int],
    slot_types: List[str],
    *,
    head_len: int,
    body_len: int,
) -> List[Dict[str, Any]]:
    """Per-slot GT/pred dump for masked SFT evaluation."""
    out: List[Dict[str, Any]] = []
    hl = int(head_len)
    bl = int(body_len)
    if hl <= 0 or bl <= 0 or hl >= len(input_ids):
        hl, bl = _skeleton_head_body_len(tokenizer, input_ids)
    end = min(len(input_ids), hl + bl, len(labels), len(pred_ids))
    si = -1
    cur_gt: List[int] = []
    cur_pred: List[int] = []

    def flush() -> None:
        nonlocal cur_gt, cur_pred
        if si < 0:
            cur_gt, cur_pred = [], []
            return
        gt = tokenizer.decode(cur_gt, skip_special_tokens=True).strip()
        pr = tokenizer.decode(cur_pred, skip_special_tokens=True).strip()
        if gt or pr:
            n = min(len(cur_gt), len(cur_pred))
            tok_acc = None
            if n:
                tok_acc = sum(1 for a, b in zip(cur_gt[:n], cur_pred[:n]) if a == b) / float(n)
            out.append({
                "slot": int(si),
                "slot_type": slot_types[si] if si < len(slot_types) else "FILL",
                "gt": gt,
                "pred": pr,
                "token_acc": tok_acc,
            })
        cur_gt, cur_pred = [], []

    for pos in range(hl, end):
        tid = int(input_ids[pos])
        if _token_is_fill_marker(tid, fill_ids, tokenizer):
            flush()
            si += 1
            continue
        if int(labels[pos]) != -100 and si >= 0:
            cur_gt.append(int(labels[pos]))
            cur_pred.append(int(pred_ids[pos]))
    flush()
    return out


@torch.inference_mode()
def _forward_one(
    model: StrucLift,
    batch: Dict[str, Any],
    *,
    alignment: str,
    fill_token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    slot_types = batch["slot_types"]

    slot_weight_mask = None
    if (labels != -100).any():
        swb = SlotWeightBuilder()
        slot_weight_mask = swb.build_slot_weights_batched_for_supervised_labels(
            input_ids,
            labels,
            slot_types,
            model.config.module_c,
            fill_token_ids=fill_token_ids,
        )

    run_align = alignment == "gt_src"
    out = model(
        bin_opcodes=batch["bin_opcodes"],
        bin_operand_types=batch["bin_operand_types"],
        bin_operand_values=batch["bin_operand_values"],
        bin_block_lengths=batch["bin_block_lengths"],
        bin_edge_index=batch["bin_edge_index"],
        bin_edge_type=batch["bin_edge_type"],
        bin_struct_features=batch["bin_struct_features"],
        bin_batch=batch.get("bin_batch"),
        src_token_ids=batch.get("src_token_ids") if run_align else None,
        src_stmt_lengths=batch.get("src_stmt_lengths") if run_align else None,
        src_edge_index=batch.get("src_edge_index") if run_align else None,
        src_edge_type=batch.get("src_edge_type") if run_align else None,
        src_ast_types=batch.get("src_ast_types") if run_align else None,
        src_compound_id=batch.get("src_compound_id") if run_align else None,
        src_batch=batch.get("src_batch") if run_align else None,
        bin_adj=batch.get("adj_bin") if run_align else None,
        src_adj=batch.get("adj_src_gt") if run_align else None,
        M_gt=batch.get("M_gt") if run_align else None,
        run_alignment=run_align,
        input_ids=input_ids,
        graph_kv=batch.get("graph_kv"),
        slot_mask=batch.get("slot_mask"),
        labels=labels,
        slot_weight_mask=slot_weight_mask,
    )
    if out.logits is None:
        raise RuntimeError("forward 未返回 logits")
    return out.logits, labels


def _encode_bin_only_cache(
    model: StrucLift,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    """二进制 encode 一次；AR 各步只重跑 decoder（避免每 token 重算 Module A）。"""
    enc = model.encode(
        bin_opcodes=batch["bin_opcodes"],
        bin_operand_types=batch["bin_operand_types"],
        bin_operand_values=batch["bin_operand_values"],
        bin_block_lengths=batch["bin_block_lengths"],
        bin_edge_index=batch["bin_edge_index"],
        bin_edge_type=batch["bin_edge_type"],
        bin_struct_features=batch["bin_struct_features"],
        bin_batch=batch.get("bin_batch"),
        run_alignment=False,
    )
    bb = batch.get("bin_batch")
    if bb is None:
        n = enc.binary_node_embs.size(0)
        bb = torch.zeros(n, dtype=torch.long, device=enc.binary_node_embs.device)
    graph_kv = batched_graph_kv_from_nodes(
        enc.binary_node_embs, bb, batch_size=1,
    )
    return {
        "bin_node_embs": enc.binary_node_embs,
        "pattern_logits": enc.pattern_logits,
        "binary_graph_emb": enc.binary_graph_emb,
        "graph_kv": graph_kv,
    }


def _forward_decode_with_bin_cache(
    model: StrucLift,
    batch: Dict[str, Any],
    enc_cache: Dict[str, Any],
) -> torch.Tensor:
    """仅 decoder；二进制侧用 :func:`_encode_bin_only_cache` 的结果。"""
    out = model(
        bin_opcodes=batch["bin_opcodes"],
        bin_operand_types=batch["bin_operand_types"],
        bin_operand_values=batch["bin_operand_values"],
        bin_block_lengths=batch["bin_block_lengths"],
        bin_edge_index=batch["bin_edge_index"],
        bin_edge_type=batch["bin_edge_type"],
        bin_struct_features=batch["bin_struct_features"],
        bin_batch=batch.get("bin_batch"),
        run_alignment=False,
        precomputed_bin_node_embs=enc_cache["bin_node_embs"],
        precomputed_pattern_logits=enc_cache["pattern_logits"],
        precomputed_bin_graph_emb=enc_cache["binary_graph_emb"],
        input_ids=batch["input_ids"],
        graph_kv=enc_cache["graph_kv"],
        slot_mask=batch.get("slot_mask"),
        labels=batch["labels"],
    )
    if out.logits is None:
        raise RuntimeError("forward 未返回 logits")
    return out.logits


@torch.inference_mode()
def _autoregressive_fill_slots(
    model: StrucLift,
    rec_infer: Dict[str, Any],
    *,
    fill_positions: List[int],
    fill_ids: List[int],
    tokenizer: Any,
    pad_id: int,
    max_slot_tokens: int,
    max_seq_len: Optional[int],
    device: torch.device,
    alignment: str,
    fill_token_ids: torch.Tensor,
    bin_only_collate: bool,
    func_name: str = "",
) -> List[int]:
    """部署推理：每个 FILL 后自回归生成 token（encoder 只算一次）。"""
    del fill_positions, alignment
    ids = [int(x) for x in rec_infer["sft_input_ids"]]
    fid = _fill_id_set(fill_ids)
    eff_max = len(ids) if max_seq_len is None else int(max_seq_len)

    st, fp_list, bb = _sync_slots_to_fills(
        ids,
        list(rec_infer.get("slot_types") or []),
        list(rec_infer.get("sft_fill_positions") or []),
        list(rec_infer.get("sft_slot_bb_indices") or []),
        fill_ids,
        tokenizer,
    )
    rec_infer["slot_types"] = st
    rec_infer["sft_fill_positions"] = fp_list
    rec_infer["sft_slot_bb_indices"] = bb
    rec_infer["sft_input_ids"] = ids
    rec_infer["sft_labels"] = [-100] * len(ids)
    batch0 = collate_binskel_stage3_batch(
        [rec_infer],
        pad_token_id=pad_id,
        max_sft_seq_len=max_seq_len,
        fill_token_ids=fill_token_ids,
        bin_only=bin_only_collate,
    )
    batch0 = move_binskel_batch_to_device(batch0, device)
    enc_cache = _encode_bin_only_cache(model, batch0)
    n_fill = len(_fill_positions_in_ids(ids, fill_ids, tokenizer))
    logger.info(
        "[%s] AR 开始: %d 槽 × 最多 %d token/槽（encoder 已缓存，每 token 仍跑 LLM 全长）",
        func_name or "?",
        n_fill,
        int(max_slot_tokens),
    )

    slot_i = 0
    while slot_i < 128:
        fills = _fill_positions_in_ids(ids, fill_ids, tokenizer)
        if slot_i >= len(fills):
            break
        fp = int(fills[slot_i])
        slot_i += 1
        pos = fp + 1
        n_tok = 0
        for _ in range(int(max_slot_tokens)):
            if pos >= eff_max:
                break
            while pos >= len(ids):
                ids.append(int(pad_id))
            rec_infer["sft_input_ids"] = ids
            rec_infer["sft_labels"] = [-100] * len(ids)
            batch = collate_binskel_stage3_batch(
                [rec_infer],
                pad_token_id=pad_id,
                max_sft_seq_len=max_seq_len,
                fill_token_ids=fill_token_ids,
                bin_only=bin_only_collate,
            )
            batch = move_binskel_batch_to_device(batch, device)
            logits = _forward_decode_with_bin_cache(model, batch, enc_cache)
            pred = int(logits[0, pos - 1].argmax(dim=-1).item())
            if _token_is_fill_marker(pred, fid, tokenizer):
                break
            ids[pos] = pred
            pos += 1
            n_tok += 1
        logger.info(
            "[%s] AR 槽 %d/%d 完成（本槽生成 %d token，序列长 %d）",
            func_name or "?",
            slot_i,
            n_fill,
            n_tok,
            len(ids),
        )
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage3 二进制函数复原评测（N 条 jsonl）")
    ap.add_argument("--init-from", type=Path, required=True, help="best_stage3.pt 等")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--llm", type=str, required=True)
    ap.add_argument("--tokenizer-saved", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--llm-device-map-auto",
        action="store_true",
        help="与训练相同：accelerate device_map=auto 切分 LLM（需多卡可见）",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--profile",
        choices=("train", "deploy", "custom"),
        default="deploy",
        help="deploy=仅二进制推理（默认）；train=调试/上界；custom=听 CLI",
    )
    ap.add_argument(
        "--decode",
        choices=("autoregressive", "one_shot"),
        default="autoregressive",
        help="deploy：autoregressive=逐槽自回归；one_shot=单次 forward+pad 槽",
    )
    ap.add_argument(
        "--metrics-from-gt",
        action="store_true",
        help="可选：用 --gt-jsonl 写 reference.c / fill_acc，**不参与**模型前向",
    )
    ap.add_argument(
        "--eval-mode",
        choices=("auto", "sft_greedy", "bin_skeleton"),
        default="auto",
        help="auto：train 下优先 jsonl 的 sft_*；否则 bin_skeleton",
    )
    ap.add_argument(
        "--skeleton-patterns",
        choices=("jsonl_gt", "module_a"),
        default="jsonl_gt",
        help="custom/deploy 时生效；train 固定 jsonl_gt",
    )
    ap.add_argument(
        "--alignment",
        choices=("gt_src", "bin_only"),
        default="gt_src",
        help="custom/deploy 时生效；train 固定 gt_src",
    )
    ap.add_argument(
        "--train-match-input",
        action="store_true",
        help="debug: keep GT tokens in model input to measure train-distribution upper bound",
    )
    ap.add_argument("--max-sft-seq-len", type=int, default=16384)
    ap.add_argument(
        "--gt-jsonl",
        type=Path,
        default=None,
        help="可选：仅当 --metrics-from-gt 时用于 reference / fill_acc（推理不用源码）",
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="解析 gt 样本 source_file 时的源码根目录（传给 augment）",
    )
    ap.add_argument(
        "--rewrite-source-prefix",
        nargs=2,
        metavar=("FROM_PREFIX", "TO_PREFIX"),
        action="append",
        dest="prefix_pairs",
        help="与 augment 相同：改写 source_file 前缀后再读盘（可多组）",
    )
    ap.add_argument("--seed", type=int, default=0, help="抽样 jsonl 的随机种子（0=取前 N 条）")
    ap.add_argument("--shuffle", action="store_true", help="随机抽 N 条而非取前 N 条")
    ap.add_argument(
        "--infer-slot-tokens",
        type=int,
        default=32,
        help="无 GT 槽文本时，每个 FILL 后在序列里预留的 pad 数，供单次 forward 贪心填槽（0=仅骨架占位符）",
    )
    args = ap.parse_args()

    t3m = _load_autoconfig()
    aug = _load_augment()
    device = torch.device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl_records(args.jsonl.expanduser().resolve(), 10**9)
    if not records:
        logger.error("jsonl 为空: %s", args.jsonl)
        sys.exit(1)
    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    gt_index: Dict[str, Dict[str, Any]] = {}
    if args.gt_jsonl is not None:
        gt_path = args.gt_jsonl.expanduser().resolve()
        if not gt_path.is_file():
            logger.error("--gt-jsonl 不存在: %s", gt_path)
            sys.exit(1)
        gt_index = _load_gt_index(gt_path)
        logger.info("已加载 gt 索引 %d 条 ← %s", len(gt_index), gt_path)

    source_root = (
        args.source_root.expanduser().resolve() if args.source_root is not None else None
    )
    prefix_pairs: Optional[List[Tuple[str, str]]] = None
    if args.prefix_pairs:
        prefix_pairs = [(str(a), str(b)) for a, b in args.prefix_pairs]

    prof = _resolve_eval_profile(args)
    eff_alignment = str(prof["alignment"])
    eff_skeleton = str(prof["skeleton_patterns"])
    logger.info(
        "评测 profile=%s → alignment=%s skeleton_patterns=%s eval_mode=%s",
        args.profile,
        eff_alignment,
        eff_skeleton,
        args.eval_mode,
    )
    if args.metrics_from_gt and not gt_index:
        logger.warning("--metrics-from-gt 但未提供 --gt-jsonl，将无 reference/fill_acc")

    from transformers import AutoConfig, AutoTokenizer

    llm_id = _resolve_llm_pretrained_source(args.llm)
    hf_cfg = AutoConfig.from_pretrained(llm_id, trust_remote_code=True)
    cfg = StrucLiftConfig()
    t3m._apply_config_dims(cfg, hf_cfg, None, None)
    msl = args.max_sft_seq_len if args.max_sft_seq_len > 0 else None
    cfg.training.s3_max_sft_seq_len = msl

    tok_path = str(args.tokenizer_saved.expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    mft = int(cfg.module_c.max_fill_tokens)
    fill_ids = _fill_token_id_list(tokenizer, mft)
    fill_token_ids_cpu = torch.tensor(fill_ids, dtype=torch.long)
    pad_id = _pad_id(tokenizer)
    target_vs = len(tokenizer)

    ckpt = args.init_from.expanduser().resolve()
    try:
        state_obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state_obj = torch.load(ckpt, map_location="cpu")
    if isinstance(state_obj, dict) and "model" in state_obj:
        state_obj = state_obj["model"]

    model = StrucLift(cfg)
    model = model.to(device)
    llm = t3m._load_hf_llm_autoconfig(
        cfg,
        llm_id,
        device,
        target_vocab_size=target_vs,
        device_map_auto=args.llm_device_map_auto,
    )
    # 必须先 inject 再 load ckpt：LoRA 在 decoder._llm 上，inject 前 state_dict 无这些键，
    # 会导致 best_stage3.pt 里已训 LoRA 全部被静默跳过（fill_acc 恒≈0 的常见原因）。
    model.decoder.inject_into_llm(llm)
    if args.llm_device_map_auto:
        t3m._align_decoder_adapters_with_llm_shards(model.decoder)
    missing, unexpected = load_state_dict_skip_shape_mismatch(
        model, state_obj, log=logger,
    )
    ckpt_lora = [
        k for k in state_obj
        if isinstance(state_obj.get(k), torch.Tensor) and "lora_" in k
    ]
    loaded_lora = [
        k for k in ckpt_lora if k in model.state_dict()
    ]
    logger.info(
        "Stage3 ckpt 载入: lora 键 %d/%d 与当前 model 对齐；missing 示例=%s",
        len(loaded_lora),
        len(ckpt_lora),
        (missing[:5] if missing else []),
    )
    if ckpt_lora and len(loaded_lora) == 0:
        logger.error(
            "checkpoint 含 %d 个 lora_ 权重但未载入任何一项——请确认已 inject LLM 且 "
            "tokenizer/LLM 与训练一致",
            len(ckpt_lora),
        )
    model.eval()
    model.binary_encoder.eval()
    model.source_encoder.eval()
    model.alignment_module.eval()

    fill_ids_dev = fill_token_ids_cpu.to(device)
    fill_lo = int(StructureConditionedDecoder.FILL_TOKEN_ID_RANGE_START)
    summaries: List[Dict[str, Any]] = []

    bin_only_collate = eff_alignment == "bin_only"
    logger.info(
        "评测: n=%d profile=%s alignment=%s bin_only_collate=%s device=%s ckpt=%s",
        len(records),
        args.profile,
        eff_alignment,
        bin_only_collate,
        device,
        ckpt,
    )
    logger.info(
        "token ids: pad=%s bos=%s eos=%s train_match_input=%s",
        pad_id,
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        bool(args.train_match_input),
    )

    for idx, rec in enumerate(records):
        func = str(rec.get("func_name", f"sample_{idx}"))
        bin_path = str(rec.get("binary", rec.get("bin_path", "")))
        tag = f"{idx:02d}_{func}"[:80].replace("/", "_")

        rec_work = _apply_source_rewrites_record(rec, prefix_pairs, aug)
        metrics_rec: Optional[Dict[str, Any]] = None
        if args.metrics_from_gt and gt_index and func in gt_index:
            metrics_rec = _merge_gt_fields(rec_work, gt_index[func])
            metrics_rec = _apply_source_rewrites_record(metrics_rec, prefix_pairs, aug)

        src_file = str((metrics_rec or rec_work).get("source_file") or "")
        src_ok = _source_readable(metrics_rec or rec_work, source_root, aug)

        if args.profile == "train":
            try:
                rec_work = _ensure_train_sft_fields(
                    rec_work,
                    aug,
                    tokenizer,
                    fill_ids,
                    source_root,
                    prefix_pairs,
                    msl,
                )
            except ValueError as e:
                logger.warning("[%s] train 对齐增广失败: %s", func, e)
                summaries.append({"index": idx, "func_name": func, "error": str(e)})
                continue
        elif not src_ok and args.metrics_from_gt:
            logger.warning(
                "[%s] metrics-from-gt 但 source 不可读（%s）",
                func,
                src_file,
            )

        mode_i = _pick_eval_mode(args, rec_work, prof)
        deploy_infer = args.profile == "deploy"
        eval_mask_id = int(rec_work.get("sft_supervised_input_mask_id") or pad_id)

        rec_infer: Dict[str, Any] = dict(rec_work)
        rec_infer["_eval_head_len"] = 0
        rec_infer["_eval_body_len"] = 0
        rec_infer["_eval_mode_effective"] = mode_i
        try:
            if mode_i == "sft_greedy":
                if "sft_input_ids" not in rec_work:
                    raise ValueError("缺少 sft_input_ids")
                sft_ids = [int(x) for x in rec_work["sft_input_ids"]]
                hl, bl = _skeleton_head_body_len(tokenizer, sft_ids)
                rec_infer["_eval_head_len"] = hl
                rec_infer["_eval_body_len"] = bl
                batch = collate_binskel_stage3_batch(
                    [rec_work],
                    pad_token_id=pad_id,
                    max_sft_seq_len=msl,
                    fill_token_ids=fill_token_ids_cpu,
                    bin_only=bin_only_collate,
                )
                ids0 = batch["input_ids"][0].tolist()
                labs0 = batch["labels"][0].tolist()
                if not args.train_match_input:
                    masked = _mask_gt_tokens_in_input(ids0, labs0, mask_id=eval_mask_id)
                    batch["input_ids"][0, : len(masked)] = torch.tensor(
                        masked, dtype=torch.long,
                    )
            else:
                pl_override: Optional[List[int]] = None
                if deploy_infer or eff_skeleton == "module_a":
                    pl_override = _pattern_labels_from_module_a(model, rec_work, device)
                has_gt = (
                    not deploy_infer
                    and bool(rec_work.get("block_src_lines"))
                    and (src_ok or bool(rec_work.get("sft_labels")))
                )
                head_len = 0
                body_len = 0
                if has_gt:
                    # 与训练同款序列布局（labels 标出槽位 GT token）；input 中把 GT 换成 pad
                    (
                        input_ids,
                        gt_labels_list,
                        slot_types,
                        fill_pos,
                        bb_idx,
                        head_len,
                        body_len,
                    ) = _build_infer_input_ids(
                        rec_work,
                        tokenizer,
                        fill_ids,
                        infer_only_body=False,
                        infer_slot_tokens=0,
                        pad_id=pad_id,
                        max_seq_len=msl,
                        aug_mod=aug,
                        pattern_labels_override=pl_override,
                        source_root=source_root,
                        prefix_rewrites=prefix_pairs,
                    )
                    if not args.train_match_input:
                        input_ids = _mask_gt_tokens_in_input(
                            input_ids, gt_labels_list, mask_id=eval_mask_id,
                        )
                else:
                    n_pad = 0
                    if not deploy_infer and args.decode == "one_shot":
                        n_pad = int(args.infer_slot_tokens)
                    elif deploy_infer and args.decode == "one_shot":
                        n_pad = int(args.infer_slot_tokens)
                    (
                        input_ids,
                        gt_labels_list,
                        slot_types,
                        fill_pos,
                        bb_idx,
                        head_len,
                        body_len,
                    ) = _build_infer_input_ids(
                        rec_work,
                        tokenizer,
                        fill_ids,
                        infer_only_body=True,
                        infer_slot_tokens=n_pad,
                        pad_id=pad_id,
                        max_seq_len=msl,
                        aug_mod=aug,
                        pattern_labels_override=pl_override,
                        source_root=source_root,
                        prefix_rewrites=prefix_pairs,
                    )
                    if deploy_infer and args.decode == "one_shot" and n_pad <= 0:
                        logger.warning("[%s] one_shot 需要 --infer-slot-tokens > 0", func)
                rec_infer.update(rec_work)
                slot_types, fill_pos, bb_idx = _sync_slots_to_fills(
                    input_ids, slot_types, fill_pos, bb_idx, fill_ids, tokenizer,
                )
                rec_infer["sft_input_ids"] = input_ids
                rec_infer["sft_labels"] = gt_labels_list
                rec_infer["slot_types"] = slot_types
                rec_infer["sft_fill_positions"] = fill_pos
                rec_infer["sft_slot_bb_indices"] = bb_idx
                rec_infer["_eval_head_len"] = head_len
                rec_infer["_eval_body_len"] = body_len
                batch = collate_binskel_stage3_batch(
                    [rec_infer],
                    pad_token_id=pad_id,
                    max_sft_seq_len=msl,
                    fill_token_ids=fill_token_ids_cpu,
                    bin_only=bin_only_collate,
                )
        except Exception as e:
            logger.warning("[%s] 跳过: %s", tag, e)
            summaries.append({
                "index": idx,
                "func_name": func,
                "error": str(e),
            })
            continue

        # collate 可能截断 sft 序列，按最终 token 里的 FILL 数对齐 slot_types
        Ls = int(batch["input_ids"].size(1))
        ids_trim = batch["input_ids"][0, :Ls].tolist()
        st0 = batch["slot_types"][0]
        fp0 = list(rec_infer.get("sft_fill_positions") or [])
        bb0 = list(rec_infer.get("sft_slot_bb_indices") or [])
        st1, fp1, bb1 = _sync_slots_to_fills(
            ids_trim, st0, fp0, bb0, fill_ids, tokenizer,
        )
        rec_infer["slot_types"] = st1
        rec_infer["sft_fill_positions"] = fp1
        rec_infer["sft_slot_bb_indices"] = bb1
        if st1 != st0 or fp1 != fp0:
            rec_infer["sft_input_ids"] = ids_trim
            sl_raw = rec_infer.get("sft_labels")
            if isinstance(sl_raw, list) and len(sl_raw) != len(ids_trim):
                rec_infer["sft_labels"] = sl_raw[: len(ids_trim)]
            batch = collate_binskel_stage3_batch(
                [rec_infer],
                pad_token_id=pad_id,
                max_sft_seq_len=msl,
                fill_token_ids=fill_token_ids_cpu,
                bin_only=bin_only_collate,
            )

        batch = move_binskel_batch_to_device(batch, device)
        hl = int(rec_infer.get("_eval_head_len", 0))
        bl = int(rec_infer.get("_eval_body_len", 0))
        fill_pos_list = list(rec_infer.get("sft_fill_positions") or [])

        if deploy_infer and args.decode == "autoregressive" and mode_i == "bin_skeleton":
            pred_list = _autoregressive_fill_slots(
                model,
                rec_infer,
                fill_positions=fill_pos_list,
                fill_ids=fill_ids,
                tokenizer=tokenizer,
                pad_id=pad_id,
                max_slot_tokens=int(args.infer_slot_tokens),
                max_seq_len=msl,
                device=device,
                alignment=eff_alignment,
                fill_token_ids=fill_token_ids_cpu,
                bin_only_collate=bin_only_collate,
                func_name=func,
            )
            hl, bl = _skeleton_head_body_len(tokenizer, pred_list)
            pred_ids = torch.tensor(pred_list, dtype=torch.long).unsqueeze(0)
        else:
            logits, labels = _forward_one(
                model,
                batch,
                alignment=eff_alignment,
                fill_token_ids=fill_ids_dev,
            )
            pred_ids = _greedy_fill_from_logits(
                batch["input_ids"], labels, logits, pad_id,
            )
            pred_list = pred_ids[0].tolist()
        pred_text = _export_recovered_skeleton_c(
            tokenizer,
            pred_list,
            fill_ids,
            head_len=hl,
            body_len=bl,
        )
        if not pred_text.strip():
            pred_text = _render_prediction_on_masked_template(
                tokenizer,
                batch["input_ids"][0].tolist(),
                pred_list,
                batch["labels"][0].tolist(),
                fill_ids,
                head_len=hl,
                body_len=bl,
            )
        pred_raw = _decode_skeleton_body(
            tokenizer, pred_list, head_len=hl, body_len=bl,
        ) if bl > 0 else ""
        slot_debug = _slot_prediction_debug(
            tokenizer,
            batch["input_ids"][0].tolist(),
            pred_list,
            batch["labels"][0].tolist(),
            fill_ids,
            list(rec_infer.get("slot_types") or []),
            head_len=hl,
            body_len=bl,
        )
        if not pred_text.strip():
            logger.warning(
                "[%s] 骨架段为空（槽位未生成或 ckpt 未学会；见 prompt_debug.txt）",
                func,
            )
        gt_text = ""
        acc: Optional[float] = None
        has_sup = False
        # train / sft_greedy：batch 里已有 sft_labels，可直接算 fill_acc（无需 --metrics-from-gt）
        batch_labels = batch.get("labels")
        if batch_labels is not None and (batch_labels != -100).any():
            plen = min(pred_ids.size(1), batch_labels.size(1))
            acc = _fill_token_accuracy(
                pred_ids[:, :plen], batch_labels[:, :plen], fill_ids_dev,
            )
            has_sup = True
            gt_text = _render_gt_from_masked_template(
                tokenizer,
                batch["input_ids"][0].tolist(),
                batch_labels[0].tolist(),
                fill_ids,
                head_len=hl,
                body_len=bl,
            )
        if args.metrics_from_gt and metrics_rec is not None:
            try:
                if metrics_rec.get("sft_input_ids") and metrics_rec.get("sft_labels"):
                    m_input = [int(x) for x in metrics_rec["sft_input_ids"]]
                    m_labels = [int(x) for x in metrics_rec["sft_labels"]]
                    m_hl, m_bl = _skeleton_head_body_len(tokenizer, m_input)
                else:
                    m_input, m_labels, _, _, _, m_hl, m_bl = _build_infer_input_ids(
                        metrics_rec,
                        tokenizer,
                        fill_ids,
                        infer_only_body=False,
                        infer_slot_tokens=0,
                        pad_id=pad_id,
                        max_seq_len=msl,
                        aug_mod=aug,
                        source_root=source_root,
                        prefix_rewrites=prefix_pairs,
                    )
                gt_text2 = _render_gt_from_masked_template(
                    tokenizer,
                    m_input,
                    m_labels,
                    fill_ids,
                    head_len=m_hl,
                    body_len=m_bl,
                )
                if gt_text2.strip():
                    gt_text = gt_text2
                m_masked = (
                    m_input
                    if args.train_match_input
                    else _mask_gt_tokens_in_input(m_input, m_labels, mask_id=eval_mask_id)
                )
                m_batch = collate_binskel_stage3_batch(
                    [{
                        **metrics_rec,
                        "sft_input_ids": m_masked,
                        "sft_labels": m_labels,
                    }],
                    pad_token_id=pad_id,
                    max_sft_seq_len=msl,
                    fill_token_ids=fill_token_ids_cpu,
                    bin_only=False,
                )
                m_batch = move_binskel_batch_to_device(m_batch, device)
                m_lab = m_batch["labels"]
                has_sup = (m_lab[0] != -100).any().item()
                if has_sup:
                    plen = min(pred_ids.size(1), m_lab.size(1))
                    acc = _fill_token_accuracy(
                        pred_ids[:, :plen], m_lab[:, :plen], fill_ids_dev,
                    )
            except Exception as e:
                logger.warning("[%s] metrics-from-gt 失败: %s", func, e)

        r_pred = float(compile_reward(pred_text))
        r_gt = float(compile_reward(gt_text)) if has_sup and gt_text.strip() else None

        sample_dir = out_dir / tag
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "meta.json").write_text(
            json.dumps(
                {
                    "func_name": func,
                    "binary": bin_path,
                    "source_file": src_file,
                    "profile": args.profile,
                    "eval_mode": rec_infer.get("_eval_mode_effective", mode_i),
                    "alignment": eff_alignment,
                    "skeleton_patterns": eff_skeleton,
                    "source_readable": src_ok,
                    "skeleton_head_len": int(hl),
                    "skeleton_body_len": int(bl),
                    "sft_masked_supervised_inputs": bool(
                        rec_infer.get("sft_masked_supervised_inputs", False)
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # 直接看模型填槽结果：recovered.c（仅骨架段 C，无 system/binary_context）
        recovered_hdr = (
            f"/* StructLift 推理产物 func={func} | 二进制→骨架→LLM填槽；非 IDA 整函数反编译 */\n\n"
        )
        recovered_body = (pred_text + "\n") if pred_text.strip() else "/* (empty skeleton body) */\n"
        (sample_dir / "recovered.c").write_text(recovered_hdr + recovered_body, encoding="utf-8")
        (sample_dir / "pred.c").write_text(recovered_body, encoding="utf-8")
        # 调试用：整段模型序列（含 system/binary_context），不要当「复原源码」看
        (sample_dir / "prompt_debug.txt").write_text(
            tokenizer.decode(pred_list, skip_special_tokens=False) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "slot_debug.json").write_text(
            json.dumps(slot_debug, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if pred_raw.strip() and pred_raw != pred_text:
            (sample_dir / "pred_with_fill_tokens.c").write_text(
                recovered_hdr + pred_raw + "\n",
                encoding="utf-8",
            )
        if args.metrics_from_gt and gt_text.strip():
            (sample_dir / "reference.c").write_text(
                "/* 可选对照：GT 槽文本（仅 --metrics-from-gt；非推理输入） */\n\n"
                + gt_text
                + "\n",
                encoding="utf-8",
            )
            (sample_dir / "gt.c").write_text(gt_text + "\n", encoding="utf-8")
        (sample_dir / "report.txt").write_text(
            "\n".join([
                f"func_name: {func}",
                f"binary: {bin_path}",
                f"source_file: {src_file}",
                f"profile: {args.profile}",
                f"eval_mode: {rec_infer.get('_eval_mode_effective', mode_i)}",
                f"alignment: {eff_alignment}",
                f"skeleton_patterns: {eff_skeleton}",
                f"source_readable: {src_ok}",
                f"decode: {args.decode}",
                f"fill_token_acc: {acc if acc is not None else 'n/a (no --metrics-from-gt)'}",
                f"compile_reward_pred: {r_pred:.4f}",
                f"compile_reward_gt: {r_gt if r_gt is not None else 'n/a'}",
                "",
                "=== pred (skeleton C) ===",
                pred_text,
                "",
                "=== gt (skeleton C) ===",
                gt_text,
            ]),
            encoding="utf-8",
        )

        row = {
            "index": idx,
            "func_name": func,
            "binary": bin_path,
            "source_file": src_file,
            "sft_masked_supervised_inputs": bool(
                rec_infer.get("sft_masked_supervised_inputs", False)
            ),
            "fill_token_acc": acc,
            "compile_reward_pred": r_pred,
            "compile_reward_gt": r_gt,
            "slot_debug": str(sample_dir / "slot_debug.json"),
            "has_gt_supervision": has_sup,
            "out_dir": str(sample_dir),
        }
        summaries.append(row)
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        r_gt_s = f"{r_gt:.3f}" if r_gt is not None else "n/a"
        logger.info(
            "[%d/%d] %s | fill_acc=%s compile_pred=%.3f | 看模型源码: %s/recovered.c",
            idx + 1,
            len(records),
            func,
            acc_s,
            r_pred,
            sample_dir,
        )

    valid = [s for s in summaries if "error" not in s]
    scored = [s for s in valid if s.get("has_gt_supervision")]
    agg = {}
    if valid:
        agg = {
            "n_ok": len(valid),
            "n_scored": len(scored),
            "mean_compile_reward_pred": sum(s["compile_reward_pred"] for s in valid) / len(valid),
        }
        if scored:
            agg["mean_fill_token_acc"] = sum(
                s["fill_token_acc"] for s in scored if s["fill_token_acc"] is not None
            ) / len(scored)
            agg["mean_compile_reward_gt"] = sum(
                s["compile_reward_gt"] for s in scored if s["compile_reward_gt"] is not None
            ) / len(scored)
    manifest = {
        "init_from": str(ckpt),
        "jsonl": str(args.jsonl),
        "profile": args.profile,
        "eval_mode_cli": args.eval_mode,
        "alignment": eff_alignment,
        "skeleton_patterns": eff_skeleton,
        "max_samples": args.max_samples,
        "aggregate": agg,
        "train_match_input": bool(args.train_match_input),
        "samples": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("汇总已写入 %s", out_dir / "summary.json")
    logger.info("每个函数直接看模型复原: %s/<func>/recovered.c", out_dir)
    if agg:
        logger.info(
            "汇总: n_ok=%d n_scored=%d compile_pred=%.3f",
            agg["n_ok"],
            agg.get("n_scored", 0),
            agg["mean_compile_reward_pred"],
        )


if __name__ == "__main__":
    main()
