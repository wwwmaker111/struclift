#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the repaired binary-only Module-B skeleton parser.

This is a v1 smoke trainer for the new Module-B direction:

    frozen/current Module A -> H_b + P_pat
    BinaryCSTParser        -> ctrl / parent / role / slot / region labels

Training labels come from ``scripts/export_cst_supervision_from_mgt.py``.
That exporter uses training-time source AST + M_gt to create teacher labels;
this trainer itself does not feed source-side graph information into B_skel.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from struclift.config import StrucLiftConfig  # noqa: E402
from struclift.data.binskel_jsonl import collate_binskel_batch, move_binskel_batch_to_device  # noqa: E402
from struclift.models.module_b_skel import (  # noqa: E402
    CTRL_IF,
    CTRL_LOOP,
    CTRL_NAME_TO_ID,
    CTRL_NONE,
    CTRL_SWITCH,
    OUTLINE_DEPTH_IGNORE,
    PARENT_IGNORE,
    ROLE_BODY,
    ROLE_COND,
    ROLE_ELSE,
    ROLE_IGNORE,
    ROLE_LOOP_BODY,
    ROLE_NAME_TO_ID,
    ROLE_NONE,
    ROLE_STMT,
    ROLE_SWITCH_BODY,
    ROLE_THEN,
    SLOT_IGNORE,
    SLOT_NAME_TO_ID,
    SLOT_NONE,
    BinaryCSTParser,
    bskel_metrics,
    build_cfg_structural_features,
    compute_bskel_loss,
    render_predicted_canonical_skeleton,
)
from struclift.models.struclift import StrucLift  # noqa: E402


FILL_SLOT_RE = re.compile(r"<FILL_(COND|BLOCK|BRANCH):slot_(\d+)>")


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        k = k.replace("._orig_mod.", ".")
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        out[k] = v
    return out


def load_stage2_model(path: Path, device: torch.device) -> StrucLift:
    cfg = StrucLiftConfig()
    model = StrucLift(cfg)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    state = normalize_state(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded A/AB ckpt = {path}", flush=True)
    print(f"missing = {len(missing)} unexpected = {len(unexpected)}", flush=True)
    if missing[:10]:
        print("missing examples:", missing[:10], flush=True)
    if unexpected[:10]:
        print("unexpected examples:", unexpected[:10], flush=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def read_jsonl_by_line(path: Path, wanted_lines: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    wanted = set(int(x) for x in wanted_lines)
    out: Dict[int, Dict[str, Any]] = {}
    if not wanted:
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no not in wanted:
                continue
            out[line_no] = json.loads(line)
            if len(out) >= len(wanted):
                break
    return out


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _slot_role_to_label(role: str) -> int:
    role = str(role or "")
    if role in ROLE_NAME_TO_ID:
        return ROLE_NAME_TO_ID[role]
    if role == "switch_body":
        return ROLE_SWITCH_BODY
    if role == "loop_body":
        return ROLE_LOOP_BODY
    if role in {"then_body", "then"}:
        return ROLE_THEN
    if role in {"else_body", "else"}:
        return ROLE_ELSE
    if role in {"stmt", "statement"}:
        return ROLE_STMT
    if role in {"body", "branch"}:
        return ROLE_BODY
    return ROLE_NONE


def _slot_type_to_label(slot_type: str) -> int:
    return int(SLOT_NAME_TO_ID.get(str(slot_type or ""), SLOT_NONE))


def _outline_targets(teacher: Dict[str, Any], *, max_controls: int, max_depth: int) -> Tuple[List[int], List[int]]:
    pairs = teacher.get("teacher_outline_pairs")
    if not isinstance(pairs, list):
        pairs = []
    raw: List[Tuple[str, int]] = []
    for x in pairs:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            kind = str(x[0])
            if kind in CTRL_NAME_TO_ID and kind != "none":
                raw.append((kind, int(x[1])))
    raw = raw[: int(max_controls)]
    if raw:
        levels = {d: i for i, d in enumerate(sorted({int(d) for _k, d in raw}))}
    else:
        levels = {}
    kinds = [CTRL_NONE] * int(max_controls)
    depths = [OUTLINE_DEPTH_IGNORE] * int(max_controls)
    for i, (kind, depth) in enumerate(raw):
        kinds[i] = int(CTRL_NAME_TO_ID.get(kind, CTRL_NONE))
        depths[i] = min(int(levels.get(int(depth), 0)), int(max_depth) - 1)
    return kinds, depths


def _slot_layout_targets(
    teacher: Dict[str, Any],
    *,
    max_items: int,
    max_depth: int,
) -> Tuple[List[int], List[int]]:
    """Teacher slot-layout sequence: FILL_BLOCK/FILL_COND/FILL_BRANCH + indent depth.

    This is deliberately not a fill-answer target.  It teaches Module B where
    slots exist in the canonical skeleton; Module C remains responsible for
    filling the slot text.
    """
    raw: List[Tuple[int, int]] = []
    teacher_skeleton = str(teacher.get("teacher_skeleton") or "")
    for line in teacher_skeleton.splitlines():
        m = FILL_SLOT_RE.search(line)
        if not m:
            continue
        try:
            sid = int(m.group(2))
        except Exception:
            sid = -1
        if sid == 0:
            continue
        slot_lab = _slot_type_to_label(f"FILL_{m.group(1)}")
        depth = (len(line) - len(line.lstrip(" "))) // 4
        raw.append((slot_lab, depth))
    if not raw:
        slots = teacher.get("slots") if isinstance(teacher.get("slots"), list) else []
        for s in slots:
            if str(s.get("role") or "") == "preamble":
                continue
            slot_lab = _slot_type_to_label(str(s.get("slot_type") or ""))
            if slot_lab == SLOT_NONE:
                continue
            try:
                depth = int(s.get("render_depth", s.get("depth", s.get("indent_depth", 1))))
            except Exception:
                depth = 1
            raw.append((slot_lab, depth))
    raw = raw[: int(max_items)]
    kinds = [SLOT_NONE] * int(max_items)
    depths = [OUTLINE_DEPTH_IGNORE] * int(max_items)
    for i, (kind, depth) in enumerate(raw):
        kinds[i] = int(kind)
        depths[i] = min(max(0, int(depth)), int(max_depth) - 1)
    return kinds, depths


def _slot_sequence_from_skeleton(teacher: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for line in str(teacher.get("teacher_skeleton") or "").splitlines():
        m = FILL_SLOT_RE.search(line)
        if not m:
            continue
        try:
            sid = int(m.group(2))
        except Exception:
            sid = -1
        if sid > 0:
            out.append(sid)
    return out


def _slot_grounding_targets(
    teacher: Dict[str, Any],
    n_bb: int,
    *,
    max_items: int,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Slot-position -> BB primary/context targets for Module-C grounding."""
    slots = teacher.get("slots") if isinstance(teacher.get("slots"), list) else []
    by_sid: Dict[int, Dict[str, Any]] = {}
    for s in slots:
        try:
            by_sid[int(s.get("slot_id"))] = s
        except Exception:
            continue
    seq = _slot_sequence_from_skeleton(teacher)
    if not seq:
        seq = [
            int(sid) for sid, s in sorted(by_sid.items())
            if int(sid) > 0 and bool(s.get("trainable", True))
        ]
    primary = [[0.0 for _ in range(n_bb)] for _ in range(int(max_items))]
    context = [[0.0 for _ in range(n_bb)] for _ in range(int(max_items))]
    mask = [[0.0 for _ in range(n_bb)] for _ in range(int(max_items))]
    for pos, sid in enumerate(seq[: int(max_items)]):
        s = by_sid.get(int(sid))
        if not s:
            continue
        p_bbs = [int(x) for x in (s.get("primary_bbs", s.get("bbs")) or []) if 0 <= int(x) < n_bb]
        c_bbs = [int(x) for x in (s.get("context_bbs", s.get("bbs")) or []) if 0 <= int(x) < n_bb]
        for bb in range(n_bb):
            mask[pos][bb] = 1.0
        for bb in p_bbs:
            primary[pos][int(bb)] = 1.0
        for bb in c_bbs:
            context[pos][int(bb)] = 1.0
    return primary, context, mask


def build_targets(
    teacher: Dict[str, Any],
    n_bb: int,
    *,
    max_outline_controls: int = 12,
    max_outline_depth: int = 8,
    max_slot_layout_items: int = 45,
    max_slot_layout_depth: int = 8,
) -> Dict[str, Any]:
    """Convert CST teacher JSON into per-BB targets for B_skel v1."""
    ctrl = [CTRL_NONE] * n_bb
    role = [ROLE_IGNORE] * n_bb
    slot = [SLOT_IGNORE] * n_bb
    parent_local = [PARENT_IGNORE] * n_bb
    region_target = [[0.0 for _ in range(n_bb)] for _ in range(n_bb)]
    same_slot_target = [[0.0 for _ in range(n_bb)] for _ in range(n_bb)]
    slot_anchor_target = [-1] * n_bb
    slot_col_mass = [1e-6 for _ in range(n_bb)]
    skip_parent_supervision: set[int] = set()
    outline_kind_target, outline_depth_target = _outline_targets(
        teacher,
        max_controls=int(max_outline_controls),
        max_depth=int(max_outline_depth),
    )
    outline_count_target = sum(1 for x in outline_kind_target if int(x) != CTRL_NONE)
    slot_layout_kind_target, slot_layout_depth_target = _slot_layout_targets(
        teacher,
        max_items=int(max_slot_layout_items),
        max_depth=int(max_slot_layout_depth),
    )
    slot_layout_count_target = sum(1 for x in slot_layout_kind_target if int(x) != SLOT_NONE)
    slot_primary_target, slot_context_target, slot_ground_mask = _slot_grounding_targets(
        teacher,
        n_bb,
        max_items=int(max_slot_layout_items),
    )

    controls = teacher.get("controls") if isinstance(teacher.get("controls"), list) else []
    slots = teacher.get("slots") if isinstance(teacher.get("slots"), list) else []

    ctrl_id_to_header: Dict[int, int] = {}
    ctrl_id_to_kind: Dict[int, str] = {}
    for c in controls:
        try:
            cid = int(c.get("control_id"))
        except Exception:
            continue
        kind = str(c.get("kind") or "none")
        cond_bbs = [int(x) for x in (c.get("condition_bbs") or []) if 0 <= int(x) < n_bb]
        region_bbs = [int(x) for x in (c.get("region_bbs") or []) if 0 <= int(x) < n_bb]
        header = cond_bbs[0] if cond_bbs else (region_bbs[0] if region_bbs else -1)
        if header < 0:
            continue
        ctrl_id_to_header[cid] = header
        ctrl_id_to_kind[cid] = kind
        ctrl[header] = int(CTRL_NAME_TO_ID.get(kind, CTRL_NONE))
        role[header] = ROLE_COND
        for bb in region_bbs:
            if 0 <= int(bb) < n_bb:
                region_target[header][int(bb)] = 1.0

    for c in controls:
        try:
            cid = int(c.get("control_id"))
        except Exception:
            continue
        header = ctrl_id_to_header.get(cid)
        if header is None:
            continue
        parent = c.get("parent_control")
        if parent is None:
            parent_local[header] = n_bb
        else:
            try:
                pheader = ctrl_id_to_header.get(int(parent))
            except Exception:
                pheader = None
            if pheader is None:
                parent_local[header] = n_bb
            elif int(pheader) == int(header):
                # AST control nodes can collapse onto the same BB after M_gt
                # projection.  That parent edge is real in source space, but
                # impossible to supervise as a per-BB parent edge.
                parent_local[header] = PARENT_IGNORE
                skip_parent_supervision.add(int(header))
            else:
                parent_local[header] = int(pheader)

    for s in slots:
        if not bool(s.get("trainable", True)):
            continue
        bbs_raw = s.get("primary_bbs", s.get("bbs"))
        bbs = [int(x) for x in (bbs_raw or []) if 0 <= int(x) < n_bb]
        if not bbs:
            continue
        slot_lab = _slot_type_to_label(str(s.get("slot_type") or ""))
        role_lab = _slot_role_to_label(str(s.get("role") or ""))
        anchor = min(bbs)
        slot_col_mass[anchor] += float(len(bbs))
        owner = s.get("owner_control")
        parent_header: Optional[int] = None
        if owner is not None:
            try:
                parent_header = ctrl_id_to_header.get(int(owner))
            except Exception:
                parent_header = None
        for bb in bbs:
            if slot[bb] == SLOT_IGNORE:
                slot[bb] = slot_lab
            if role[bb] == ROLE_IGNORE or role[bb] == ROLE_NONE:
                role[bb] = role_lab
            if parent_local[bb] == PARENT_IGNORE and int(bb) not in skip_parent_supervision:
                parent_local[bb] = int(parent_header) if parent_header is not None else n_bb
            slot_anchor_target[bb] = anchor
        for a in bbs:
            for b in bbs:
                same_slot_target[int(a)][int(b)] = 1.0

    for i in range(n_bb):
        if role[i] == ROLE_IGNORE:
            role[i] = ROLE_NONE
        if slot[i] == SLOT_IGNORE:
            slot[i] = SLOT_NONE
        if parent_local[i] == PARENT_IGNORE and i not in skip_parent_supervision:
            parent_local[i] = n_bb
        if slot_anchor_target[i] < 0:
            slot_anchor_target[i] = i
            slot_col_mass[i] += 1.0
    return {
        "bskel_ctrl_targets": ctrl,
        "bskel_role_targets": role,
        "bskel_slot_targets": slot,
        "bskel_parent_targets": parent_local,
        "bskel_region_targets": region_target,
        "bskel_same_slot_targets": same_slot_target,
        "bskel_slot_anchor_targets": slot_anchor_target,
        "bskel_slot_col_mass": slot_col_mass,
        "bskel_outline_count_targets": outline_count_target,
        "bskel_outline_kind_targets": outline_kind_target,
        "bskel_outline_depth_targets": outline_depth_target,
        "bskel_slot_layout_count_targets": slot_layout_count_target,
        "bskel_slot_layout_kind_targets": slot_layout_kind_target,
        "bskel_slot_layout_depth_targets": slot_layout_depth_target,
        "bskel_slot_primary_targets": slot_primary_target,
        "bskel_slot_context_targets": slot_context_target,
        "bskel_slot_grounding_mask": slot_ground_mask,
    }


@dataclass
class BSkelItem:
    row: Dict[str, Any]
    teacher: Dict[str, Any]
    cached_binary_node_embs: Optional[torch.Tensor] = None
    cached_pattern_logits: Optional[torch.Tensor] = None
    cached_cfg_features: Optional[Dict[str, torch.Tensor]] = None


class BSkelTeacherDataset(Dataset):
    def __init__(
        self,
        source_jsonl: Path,
        teacher_jsonl: Path,
        *,
        max_examples: int = 0,
        max_outline_controls: int = 12,
        max_outline_depth: int = 8,
        max_slot_layout_items: int = 45,
        max_slot_layout_depth: int = 8,
    ) -> None:
        self.source_jsonl = source_jsonl.expanduser().resolve()
        self.teacher_jsonl = teacher_jsonl.expanduser().resolve()
        teachers = list(iter_jsonl(self.teacher_jsonl))
        if max_examples and max_examples > 0:
            teachers = teachers[: int(max_examples)]
        wanted = [int(t["input_line"]) for t in teachers]
        source_by_line = read_jsonl_by_line(self.source_jsonl, wanted)
        self.items: List[BSkelItem] = []
        for t in teachers:
            line_no = int(t.get("input_line") or 0)
            row = source_by_line.get(line_no)
            if row is None:
                continue
            n_bb = len(row.get("bin_block_lengths") or [])
            targets = build_targets(
                t,
                n_bb,
                max_outline_controls=int(max_outline_controls),
                max_outline_depth=int(max_outline_depth),
                max_slot_layout_items=int(max_slot_layout_items),
                max_slot_layout_depth=int(max_slot_layout_depth),
            )
            merged = dict(row)
            merged.update(targets)
            self.items.append(BSkelItem(row=merged, teacher=t))
        if not self.items:
            raise RuntimeError(f"no usable samples from {teacher_jsonl}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        row = dict(item.row)
        if item.cached_binary_node_embs is not None and item.cached_pattern_logits is not None:
            row["_bskel_cached_binary_node_embs"] = item.cached_binary_node_embs
            row["_bskel_cached_pattern_logits"] = item.cached_pattern_logits
        if item.cached_cfg_features is not None:
            row["_bskel_cached_cfg_features"] = item.cached_cfg_features
        return row


def _block_diag_2d(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    if not parts:
        return torch.zeros(0, 0)
    rows = sum(int(p.size(0)) for p in parts)
    cols = sum(int(p.size(1)) for p in parts)
    out = parts[0].new_zeros(rows, cols)
    r = 0
    c = 0
    for p in parts:
        nr = int(p.size(0))
        nc = int(p.size(1))
        out[r : r + nr, c : c + nc] = p
        r += nr
        c += nc
    return out


def collate_bskel(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    base = collate_binskel_batch(samples, inst_dropout_p=0.0, edge_dropout_p=0.0, add_func_uid=False)
    ctrl: List[int] = []
    role: List[int] = []
    slot: List[int] = []
    parents: List[List[int]] = []
    regions: List[torch.Tensor] = []
    same_slots: List[torch.Tensor] = []
    slot_anchors: List[List[int]] = []
    slot_col_mass: List[torch.Tensor] = []
    outline_kinds: List[List[int]] = []
    outline_depths: List[List[int]] = []
    outline_counts: List[int] = []
    slot_layout_kinds: List[List[int]] = []
    slot_layout_depths: List[List[int]] = []
    slot_layout_counts: List[int] = []
    slot_primary_targets: List[torch.Tensor] = []
    slot_context_targets: List[torch.Tensor] = []
    slot_grounding_masks: List[torch.Tensor] = []
    cached_node_embs: List[torch.Tensor] = []
    cached_pattern_logits: List[torch.Tensor] = []
    cached_cfg_rows: List[Dict[str, torch.Tensor]] = []
    has_encoder_cache = all(
        "_bskel_cached_binary_node_embs" in s and "_bskel_cached_pattern_logits" in s
        for s in samples
    )
    has_cfg_cache = all("_bskel_cached_cfg_features" in s for s in samples)
    for s in samples:
        ctrl.extend(int(x) for x in s["bskel_ctrl_targets"])
        role.extend(int(x) for x in s["bskel_role_targets"])
        slot.extend(int(x) for x in s["bskel_slot_targets"])
        parents.append([int(x) for x in s["bskel_parent_targets"]])
        regions.append(torch.tensor(s["bskel_region_targets"], dtype=torch.float32))
        same_slots.append(torch.tensor(s["bskel_same_slot_targets"], dtype=torch.float32))
        slot_anchors.append([int(x) for x in s["bskel_slot_anchor_targets"]])
        slot_col_mass.append(torch.tensor(s["bskel_slot_col_mass"], dtype=torch.float32))
        outline_kinds.append([int(x) for x in s["bskel_outline_kind_targets"]])
        outline_depths.append([int(x) for x in s["bskel_outline_depth_targets"]])
        outline_counts.append(int(s["bskel_outline_count_targets"]))
        slot_layout_kinds.append([int(x) for x in s["bskel_slot_layout_kind_targets"]])
        slot_layout_depths.append([int(x) for x in s["bskel_slot_layout_depth_targets"]])
        slot_layout_counts.append(int(s["bskel_slot_layout_count_targets"]))
        slot_primary_targets.append(torch.tensor(s["bskel_slot_primary_targets"], dtype=torch.float32))
        slot_context_targets.append(torch.tensor(s["bskel_slot_context_targets"], dtype=torch.float32))
        slot_grounding_masks.append(torch.tensor(s["bskel_slot_grounding_mask"], dtype=torch.float32))
        if has_encoder_cache:
            cached_node_embs.append(s["_bskel_cached_binary_node_embs"])
            cached_pattern_logits.append(s["_bskel_cached_pattern_logits"])
        if has_cfg_cache:
            cached_cfg_rows.append(s["_bskel_cached_cfg_features"])
    base["bskel_ctrl_targets"] = torch.tensor(ctrl, dtype=torch.long)
    base["bskel_role_targets"] = torch.tensor(role, dtype=torch.long)
    base["bskel_slot_targets"] = torch.tensor(slot, dtype=torch.long)
    base["bskel_parent_targets_by_graph"] = parents
    base["bskel_region_targets_by_graph"] = regions
    base["bskel_same_slot_targets_by_graph"] = same_slots
    base["bskel_slot_anchor_targets_by_graph"] = slot_anchors
    base["bskel_slot_col_mass_by_graph"] = slot_col_mass
    base["bskel_outline_kind_targets"] = torch.tensor(outline_kinds, dtype=torch.long)
    base["bskel_outline_depth_targets"] = torch.tensor(outline_depths, dtype=torch.long)
    base["bskel_outline_count_targets"] = torch.tensor(outline_counts, dtype=torch.long)
    base["bskel_slot_layout_kind_targets"] = torch.tensor(slot_layout_kinds, dtype=torch.long)
    base["bskel_slot_layout_depth_targets"] = torch.tensor(slot_layout_depths, dtype=torch.long)
    base["bskel_slot_layout_count_targets"] = torch.tensor(slot_layout_counts, dtype=torch.long)
    base["bskel_slot_primary_targets_by_graph"] = slot_primary_targets
    base["bskel_slot_context_targets_by_graph"] = slot_context_targets
    base["bskel_slot_grounding_masks_by_graph"] = slot_grounding_masks
    if has_encoder_cache:
        base["bskel_cached_binary_node_embs"] = torch.cat(cached_node_embs, dim=0)
        base["bskel_cached_pattern_logits"] = torch.cat(cached_pattern_logits, dim=0)
    if has_cfg_cache:
        keys = cached_cfg_rows[0].keys() if cached_cfg_rows else []
        base["bskel_cached_cfg_features"] = {
            k: _block_diag_2d([row[k] for row in cached_cfg_rows]) for k in keys
        }
    return base


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r})
    return {k: sum(float(r.get(k, 0.0)) for r in rows) / len(rows) for k in keys}


def move_bskel_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> Dict[str, Any]:
    """Move only tensors needed by the current B_skel path to GPU.

    When frozen Module-A cache is available, opcode/operand tensors are no
    longer used inside the training step, so leaving them on CPU avoids a large
    repeated host-to-device copy.
    """
    has_cache = "bskel_cached_binary_node_embs" in batch and "bskel_cached_pattern_logits" in batch
    keep_cpu_when_cached = {
        "bin_opcodes",
        "bin_operand_types",
        "bin_operand_values",
        "bin_block_lengths",
    }
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if has_cache and k in keep_cpu_when_cached:
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device, non_blocking=non_blocking) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


@torch.no_grad()
def cache_frozen_binary_features(
    *,
    a_model: StrucLift,
    dataset: BSkelTeacherDataset,
    device: torch.device,
    batch_size: int,
    label: str,
) -> None:
    """Cache frozen Module-A outputs and fixed CFG relations for one dataset.

    Module A is frozen in this trainer, and CFG structural relations are
    deterministic.  Caching them keeps the algorithm and losses unchanged while
    avoiding the same work in every epoch.
    """
    t0 = time.time()
    a_model.eval()
    total_graphs = len(dataset)
    print(f"caching {label}: graphs={total_graphs} batch_size={batch_size}", flush=True)
    for start in range(0, total_graphs, max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), total_graphs)
        samples = [dict(dataset.items[i].row) for i in range(start, end)]
        batch = collate_bskel(samples)
        batch = move_binskel_batch_to_device(batch, device, non_blocking=True)
        enc = a_model.encode(
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
        bin_batch = batch["bin_batch"]
        for local_gi, item_idx in enumerate(range(start, end)):
            nodes = (bin_batch == local_gi).nonzero(as_tuple=True)[0]
            item = dataset.items[item_idx]
            item.cached_binary_node_embs = enc.binary_node_embs[nodes].detach().cpu().contiguous()
            item.cached_pattern_logits = enc.pattern_logits[nodes].detach().cpu().contiguous()

            row = item.row
            n_bb = len(row.get("bin_block_lengths") or [])
            raw_edge_index = row.get("bin_edge_index") or [[], []]
            raw_edge_type = row.get("bin_edge_type") or []
            edge_index = torch.tensor(raw_edge_index, dtype=torch.long)
            if edge_index.numel() == 0:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.tensor(raw_edge_type, dtype=torch.long)
            item.cached_cfg_features = {
                k: v.detach().cpu().contiguous()
                for k, v in build_cfg_structural_features(
                    edge_index,
                    edge_type,
                    n_bb,
                    max_dist=8,
                    bin_batch=torch.zeros(n_bb, dtype=torch.long),
                ).items()
            }
        print(f"cached {label}: {end}/{total_graphs}", flush=True)
    print(f"cached {label} elapsed_sec={time.time() - t0:.1f}", flush=True)


def run_epoch(
    *,
    a_model: StrucLift,
    parser: BinaryCSTParser,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    train: bool,
) -> Dict[str, float]:
    parser.train(train)
    rows: List[Dict[str, float]] = []
    for step, batch in enumerate(loader, 1):
        batch = move_bskel_batch_to_device(batch, device, non_blocking=True)
        if "bskel_cached_binary_node_embs" in batch and "bskel_cached_pattern_logits" in batch:
            binary_node_embs = batch["bskel_cached_binary_node_embs"]
            pattern_logits = batch["bskel_cached_pattern_logits"]
        else:
            with torch.no_grad():
                enc = a_model.encode(
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
            binary_node_embs = enc.binary_node_embs
            pattern_logits = enc.pattern_logits
        out = parser(
            binary_node_embs=binary_node_embs,
            pattern_logits=pattern_logits,
            bin_struct_features=batch["bin_struct_features"],
            edge_index=batch["bin_edge_index"],
            edge_type=batch["bin_edge_type"],
            bin_batch=batch.get("bin_batch"),
            cfg_features=batch.get("bskel_cached_cfg_features"),
        )
        losses = compute_bskel_loss(
            out,
            ctrl_targets=batch["bskel_ctrl_targets"],
            role_targets=batch["bskel_role_targets"],
            slot_targets=batch["bskel_slot_targets"],
            outline_count_targets=batch["bskel_outline_count_targets"],
            outline_kind_targets=batch["bskel_outline_kind_targets"],
            outline_depth_targets=batch["bskel_outline_depth_targets"],
            slot_layout_count_targets=batch["bskel_slot_layout_count_targets"],
            slot_layout_kind_targets=batch["bskel_slot_layout_kind_targets"],
            slot_layout_depth_targets=batch["bskel_slot_layout_depth_targets"],
            slot_primary_targets_by_graph=batch["bskel_slot_primary_targets_by_graph"],
            slot_context_targets_by_graph=batch["bskel_slot_context_targets_by_graph"],
            slot_grounding_masks_by_graph=batch["bskel_slot_grounding_masks_by_graph"],
            parent_targets_by_graph=batch["bskel_parent_targets_by_graph"],
            slot_anchor_targets_by_graph=batch["bskel_slot_anchor_targets_by_graph"],
            slot_col_mass_by_graph=batch["bskel_slot_col_mass_by_graph"],
            region_targets_by_graph=batch["bskel_region_targets_by_graph"],
            same_slot_targets_by_graph=batch["bskel_same_slot_targets_by_graph"],
            bin_batch=batch["bin_batch"],
            ctrl_weight=float(args.ctrl_weight),
            role_weight=float(args.role_weight),
            slot_weight=float(args.slot_weight),
            parent_weight=float(args.parent_weight),
            tree_weight=float(args.tree_weight),
            region_weight=float(args.region_weight),
            same_slot_weight=float(args.same_slot_weight),
            slot_ot_weight=float(args.slot_ot_weight),
            role_pair_weight=float(args.role_pair_weight),
            outline_count_weight=float(args.outline_count_weight),
            outline_kind_weight=float(args.outline_kind_weight),
            outline_depth_weight=float(args.outline_depth_weight),
            slot_layout_count_weight=float(args.slot_layout_count_weight),
            slot_layout_kind_weight=float(args.slot_layout_kind_weight),
            slot_layout_depth_weight=float(args.slot_layout_depth_weight),
            slot_primary_weight=float(args.slot_primary_weight),
            slot_context_weight=float(args.slot_context_weight),
            loss_profile=str(args.loss_profile),
            cst_weight=float(args.cst_weight),
            region_struct_weight=float(args.region_struct_weight),
            slot_struct_weight=float(args.slot_struct_weight),
            ot_weight=float(args.ot_weight),
            skel_weight=float(args.skel_weight),
        )
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), float(args.grad_clip))
            optimizer.step()
        metrics = bskel_metrics(
            out,
            ctrl_targets=batch["bskel_ctrl_targets"],
            role_targets=batch["bskel_role_targets"],
            slot_targets=batch["bskel_slot_targets"],
            outline_count_targets=batch["bskel_outline_count_targets"],
            outline_kind_targets=batch["bskel_outline_kind_targets"],
            outline_depth_targets=batch["bskel_outline_depth_targets"],
            slot_layout_count_targets=batch["bskel_slot_layout_count_targets"],
            slot_layout_kind_targets=batch["bskel_slot_layout_kind_targets"],
            slot_layout_depth_targets=batch["bskel_slot_layout_depth_targets"],
            slot_primary_targets_by_graph=batch["bskel_slot_primary_targets_by_graph"],
            slot_context_targets_by_graph=batch["bskel_slot_context_targets_by_graph"],
            slot_grounding_masks_by_graph=batch["bskel_slot_grounding_masks_by_graph"],
            parent_targets_by_graph=batch["bskel_parent_targets_by_graph"],
            slot_anchor_targets_by_graph=batch["bskel_slot_anchor_targets_by_graph"],
            slot_col_mass_by_graph=batch["bskel_slot_col_mass_by_graph"],
            region_targets_by_graph=batch["bskel_region_targets_by_graph"],
            same_slot_targets_by_graph=batch["bskel_same_slot_targets_by_graph"],
            bin_batch=batch["bin_batch"],
        )
        rec: Dict[str, float] = {k: float(v.detach().item()) for k, v in losses.items()}
        rec.update(metrics)
        rows.append(rec)
        if step % int(args.log_every) == 0:
            tag = "train" if train else "val"
            cur = mean_dict(rows[-int(args.log_every):])
            print(
                f"[{tag}] step={step} "
                f"loss={cur.get('loss_total',0):.4f} "
                f"cst={cur.get('loss_cst',0):.3f} "
                f"regL={cur.get('loss_region_struct',0):.3f} "
                f"slotL={cur.get('loss_slot_struct',0):.3f} "
                f"ot={cur.get('loss_ot',0):.3f} "
                f"skel={cur.get('loss_skel',0):.3f} "
                f"ctrl={cur.get('ctrl_acc',0):.3f} "
                f"parent={cur.get('parent_acc',0):.3f} "
                f"role={cur.get('role_acc',0):.3f} "
                f"slot={cur.get('slot_acc',0):.3f} "
                f"slotA={cur.get('slot_anchor_acc',0):.3f} "
                f"outline={cur.get('outline_exact_acc',0):.3f} "
                f"count={cur.get('outline_count_acc',0):.3f} "
                f"layout={cur.get('slot_layout_exact_acc',0):.3f} "
                f"layseq={cur.get('slot_layout_kindseq_acc',0):.3f} "
                f"slotP={cur.get('slot_primary_exact_acc',0):.3f} "
                f"slotC={cur.get('slot_context_exact_acc',0):.3f} "
                f"region={cur.get('region_pair_acc',0):.3f}",
                flush=True,
            )
    return mean_dict(rows)


@torch.no_grad()
def export_predictions(
    *,
    a_model: StrucLift,
    parser: BinaryCSTParser,
    dataset: BSkelTeacherDataset,
    out_path: Path,
    device: torch.device,
    num_examples: int,
) -> None:
    parser.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(int(num_examples), len(dataset))
    with out_path.open("w", encoding="utf-8") as wf:
        wf.write("===== B_SKEL V1 PREDICTED CANONICAL SKELETON SMOKE =====\n")
        wf.write(f"num_examples = {n}\n\n")
        for idx in range(n):
            sample = dataset[idx]
            batch = collate_bskel([sample])
            batch = move_bskel_batch_to_device(batch, device, non_blocking=True)
            if "bskel_cached_binary_node_embs" in batch and "bskel_cached_pattern_logits" in batch:
                binary_node_embs = batch["bskel_cached_binary_node_embs"]
                pattern_logits = batch["bskel_cached_pattern_logits"]
            else:
                enc = a_model.encode(
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
                binary_node_embs = enc.binary_node_embs
                pattern_logits = enc.pattern_logits
            pred = parser(
                binary_node_embs=binary_node_embs,
                pattern_logits=pattern_logits,
                bin_struct_features=batch["bin_struct_features"],
                edge_index=batch["bin_edge_index"],
                edge_type=batch["bin_edge_type"],
                bin_batch=batch.get("bin_batch"),
                cfg_features=batch.get("bskel_cached_cfg_features"),
            )
            metrics = bskel_metrics(
                pred,
                ctrl_targets=batch["bskel_ctrl_targets"],
                role_targets=batch["bskel_role_targets"],
                slot_targets=batch["bskel_slot_targets"],
                outline_count_targets=batch["bskel_outline_count_targets"],
                outline_kind_targets=batch["bskel_outline_kind_targets"],
                outline_depth_targets=batch["bskel_outline_depth_targets"],
                slot_layout_count_targets=batch["bskel_slot_layout_count_targets"],
                slot_layout_kind_targets=batch["bskel_slot_layout_kind_targets"],
                slot_layout_depth_targets=batch["bskel_slot_layout_depth_targets"],
                slot_primary_targets_by_graph=batch["bskel_slot_primary_targets_by_graph"],
                slot_context_targets_by_graph=batch["bskel_slot_context_targets_by_graph"],
                slot_grounding_masks_by_graph=batch["bskel_slot_grounding_masks_by_graph"],
                parent_targets_by_graph=batch["bskel_parent_targets_by_graph"],
                slot_anchor_targets_by_graph=batch["bskel_slot_anchor_targets_by_graph"],
                slot_col_mass_by_graph=batch["bskel_slot_col_mass_by_graph"],
                region_targets_by_graph=batch["bskel_region_targets_by_graph"],
                same_slot_targets_by_graph=batch["bskel_same_slot_targets_by_graph"],
                bin_batch=batch["bin_batch"],
            )
            teacher = dataset.items[idx].teacher
            wf.write("=" * 120 + "\n")
            wf.write(
                f"example={idx} input_line={teacher.get('input_line')} "
                f"func={teacher.get('func_name')} n_bb={teacher.get('n_bb')}\n"
            )
            wf.write(f"metrics={metrics}\n")
            wf.write("\n-- TEACHER CANONICAL SKELETON --\n")
            wf.write(str(teacher.get("teacher_skeleton") or "").rstrip() + "\n")
            wf.write("\n-- PREDICTED CANONICAL SKELETON --\n")
            wf.write(render_predicted_canonical_skeleton(pred).rstrip() + "\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train binary-only Module-B skeleton parser v1.")
    ap.add_argument("--train-jsonl", type=Path, required=True, help="original train JSONL with binary fields")
    ap.add_argument("--train-teacher-jsonl", type=Path, required=True, help="teacher JSONL from export_cst_supervision_from_mgt.py")
    ap.add_argument("--val-jsonl", type=Path)
    ap.add_argument("--val-teacher-jsonl", type=Path)
    ap.add_argument("--a-ckpt", type=Path, required=True, help="current A/AB checkpoint")
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-a-features", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cache-batch-size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--max-outline-controls", type=int, default=12)
    ap.add_argument("--max-outline-depth", type=int, default=8)
    ap.add_argument("--max-slot-layout-items", type=int, default=45)
    ap.add_argument("--max-slot-layout-depth", type=int, default=8)
    ap.add_argument("--max-train-examples", type=int, default=0)
    ap.add_argument("--max-val-examples", type=int, default=0)
    ap.add_argument("--ctrl-weight", type=float, default=1.0)
    ap.add_argument("--role-weight", type=float, default=1.0)
    ap.add_argument("--slot-weight", type=float, default=0.5)
    ap.add_argument("--parent-weight", type=float, default=1.0)
    ap.add_argument("--tree-weight", type=float, default=0.2)
    ap.add_argument("--region-weight", type=float, default=1.0)
    ap.add_argument("--same-slot-weight", type=float, default=0.5)
    ap.add_argument("--slot-ot-weight", type=float, default=0.5)
    ap.add_argument("--role-pair-weight", type=float, default=0.5)
    ap.add_argument("--outline-count-weight", type=float, default=4.0)
    ap.add_argument("--outline-kind-weight", type=float, default=4.0)
    ap.add_argument("--outline-depth-weight", type=float, default=2.0)
    ap.add_argument("--slot-layout-count-weight", type=float, default=6.0)
    ap.add_argument("--slot-layout-kind-weight", type=float, default=6.0)
    ap.add_argument("--slot-layout-depth-weight", type=float, default=4.0)
    ap.add_argument("--slot-primary-weight", type=float, default=2.0)
    ap.add_argument("--slot-context-weight", type=float, default=1.0)
    ap.add_argument("--loss-profile", choices=["legacy", "grouped5"], default="legacy")
    ap.add_argument("--cst-weight", type=float, default=1.0)
    ap.add_argument("--region-struct-weight", type=float, default=1.0)
    ap.add_argument("--slot-struct-weight", type=float, default=1.0)
    ap.add_argument("--ot-weight", type=float, default=1.0)
    ap.add_argument("--skel-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--export-predictions", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.save_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    print("device =", device, flush=True)
    print("save_dir =", args.save_dir.expanduser().resolve(), flush=True)

    a_model = load_stage2_model(args.a_ckpt.expanduser().resolve(), device)
    parser = BinaryCSTParser(
        d_model=StrucLiftConfig().module_b.d_model,
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        max_outline_controls=int(args.max_outline_controls),
        max_outline_depth=int(args.max_outline_depth),
        max_slot_layout_items=int(args.max_slot_layout_items),
        max_slot_layout_depth=int(args.max_slot_layout_depth),
    ).to(device)
    optimizer = torch.optim.AdamW(parser.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_ds = BSkelTeacherDataset(
        args.train_jsonl,
        args.train_teacher_jsonl,
        max_examples=int(args.max_train_examples),
        max_outline_controls=int(args.max_outline_controls),
        max_outline_depth=int(args.max_outline_depth),
        max_slot_layout_items=int(args.max_slot_layout_items),
        max_slot_layout_depth=int(args.max_slot_layout_depth),
    )
    if bool(args.cache_a_features):
        cache_frozen_binary_features(
            a_model=a_model,
            dataset=train_ds,
            device=device,
            batch_size=int(args.cache_batch_size),
            label="train",
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_bskel,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )
    val_loader = None
    if args.val_jsonl is not None and args.val_teacher_jsonl is not None:
        val_ds = BSkelTeacherDataset(
            args.val_jsonl,
            args.val_teacher_jsonl,
            max_examples=int(args.max_val_examples),
            max_outline_controls=int(args.max_outline_controls),
            max_outline_depth=int(args.max_outline_depth),
            max_slot_layout_items=int(args.max_slot_layout_items),
            max_slot_layout_depth=int(args.max_slot_layout_depth),
        )
        if bool(args.cache_a_features):
            cache_frozen_binary_features(
                a_model=a_model,
                dataset=val_ds,
                device=device,
                batch_size=int(args.cache_batch_size),
                label="val",
            )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_bskel,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.num_workers) > 0),
        )
        print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)
    else:
        print(f"train_samples={len(train_ds)} val_samples=0", flush=True)

    best_score = -1.0
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        print(f"\n===== epoch {epoch}/{args.epochs} =====", flush=True)
        train_m = run_epoch(
            a_model=a_model,
            parser=parser,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            args=args,
            train=True,
        )
        print("train_metrics =", json.dumps(train_m, sort_keys=True), flush=True)
        val_m: Dict[str, float] = {}
        if val_loader is not None:
            with torch.no_grad():
                val_m = run_epoch(
                    a_model=a_model,
                    parser=parser,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    args=args,
                    train=False,
                )
            print("val_metrics =", json.dumps(val_m, sort_keys=True), flush=True)
        score_src = val_m if val_m else train_m
        score = (
            2.0 * score_src.get("slot_layout_exact_acc", 0.0)
            + 1.5 * score_src.get("outline_exact_acc", 0.0)
            + 1.0 * score_src.get("slot_layout_kindseq_acc", 0.0)
            + 1.0 * score_src.get("slot_primary_exact_acc", 0.0)
            + 0.8 * score_src.get("slot_context_exact_acc", 0.0)
            + 1.0 * score_src.get("slot_acc", 0.0)
            + 1.0 * score_src.get("slot_anchor_acc", 0.0)
            + 0.8 * score_src.get("role_acc", 0.0)
            + 0.5 * score_src.get("parent_acc", 0.0)
            + 0.5 * score_src.get("region_pair_acc", 0.0)
        )
        ckpt = {
            "parser": parser.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "train_metrics": train_m,
            "val_metrics": val_m,
        }
        torch.save(ckpt, args.save_dir / "latest_bskel.pt")
        if score > best_score:
            best_score = score
            torch.save(ckpt, args.save_dir / "best_bskel.pt")
            print(f"saved best_bskel.pt score={best_score:.4f}", flush=True)

    print(f"elapsed_sec={time.time() - t0:.1f}", flush=True)
    print("saved_latest =", args.save_dir / "latest_bskel.pt", flush=True)
    print("saved_best =", args.save_dir / "best_bskel.pt", flush=True)
    export_ds = val_ds if val_loader is not None else train_ds  # type: ignore[name-defined]
    export_predictions(
        a_model=a_model,
        parser=parser,
        dataset=export_ds,
        out_path=args.save_dir / "bskel_v1_prediction_preview.txt",
        device=device,
        num_examples=int(args.export_predictions),
    )
    print("prediction_preview =", args.save_dir / "bskel_v1_prediction_preview.txt", flush=True)


if __name__ == "__main__":
    main()
