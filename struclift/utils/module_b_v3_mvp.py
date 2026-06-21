"""Safe Module-B v3 MVP: CFG-only candidate skeletons with validation.

This module intentionally does not depend on Module A, checkpoints, source ASTs,
or neural predictions.  It produces a conservative, auditable skeleton from a
binary CFG and verifies the hard invariants needed before Module C can consume
the result:

* every BB is covered by a primary slot or fallback region;
* every CFG edge is either represented by an explicit structured semantic edge,
  a linear slot edge, a fallback region, or an explicit GOTO slot;
* every fillable slot carries primary/context BB grounding;
* the rendered skeleton is syntactically balanced.

The first version is a safety layer.  It prefers readable structured regions
when the CFG shape is simple, but falls back to explicit unstructured/goto slots
instead of inventing unsupported source-like control flow.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


Edge = Tuple[int, int]


@dataclass
class CFGView:
    n_bb: int
    edges: List[Edge]
    succs: Dict[int, List[int]]
    preds: Dict[int, List[int]]
    dfs_order: List[int]
    unreachable: List[int]
    dom: List[Set[int]]
    idom: List[int]
    pdom: List[Set[int]]
    ipdom: List[int]
    sccs: List[Set[int]]
    irreducible_sccs: List[Set[int]]
    back_edges: List[Edge]
    loop_nodes_by_header: Dict[int, Set[int]]
    sese_regions: List[Dict[str, Any]]


@dataclass
class Slot:
    slot_id: int
    slot_type: str
    primary_bbs: List[int]
    context_bbs: List[int]
    role: str = ""
    owner: Optional[str] = None
    edge: Optional[Edge] = None
    reason: str = ""

    def asdict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "slot_id": int(self.slot_id),
            "slot_type": self.slot_type,
            "role": self.role,
            "primary_bbs": list(self.primary_bbs),
            "context_bbs": list(self.context_bbs),
        }
        if self.owner is not None:
            out["owner"] = self.owner
        if self.edge is not None:
            out["edge"] = [int(self.edge[0]), int(self.edge[1])]
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class Node:
    kind: str
    slot: Optional[Slot] = None
    children: List["Node"] = field(default_factory=list)
    else_children: List["Node"] = field(default_factory=list)
    region_bbs: List[int] = field(default_factory=list)
    reason: str = ""

    def asdict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "region_bbs": list(self.region_bbs),
        }
        if self.slot is not None:
            out["slot"] = self.slot.asdict()
        if self.children:
            out["children"] = [c.asdict() for c in self.children]
        if self.else_children:
            out["else_children"] = [c.asdict() for c in self.else_children]
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class Candidate:
    name: str
    root: Node
    slots: List[Slot]
    explicit_edges: Set[Edge]
    structured_regions: List[Set[int]]
    fallback_regions: List[Set[int]]
    interval_regions: List[Set[int]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    valid: bool
    bb_coverage_ok: bool
    unique_primary_ok: bool
    edge_preservation_ok: bool
    slot_grounding_ok: bool
    parse_success: bool
    fatal_errors: List[str]
    missing_bbs: List[int]
    duplicate_primary_bbs: Dict[int, List[int]]
    uncovered_edges: List[Edge]
    invalid_bb_refs: List[Dict[str, Any]]
    edge_explanation_counts: Dict[str, int]
    edge_explanation_preview: Dict[str, List[List[int]]]
    strict_edge_preservation_ok: bool
    fallback_region_count: int
    fallback_bb_count: int
    goto_count: int
    break_count: int
    continue_count: int
    early_exit_count: int
    cleanup_exit_count: int
    escape_count: int
    structured_control_count: int
    structured_bb_count: int
    slot_count: int
    slot_locality_score: float
    empty_control_count: int
    empty_loop_count: int
    empty_if_count: int
    empty_switch_count: int
    empty_control_with_context_count: int
    control_body_total_bb_count: int
    control_body_nested_bb_count: int
    body_bb_outside_control_count: int
    score: float

    def asdict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "bb_coverage_ok": self.bb_coverage_ok,
            "unique_primary_ok": self.unique_primary_ok,
            "edge_preservation_ok": self.edge_preservation_ok,
            "slot_grounding_ok": self.slot_grounding_ok,
            "parse_success": self.parse_success,
            "fatal_errors": list(self.fatal_errors),
            "missing_bbs": list(self.missing_bbs),
            "duplicate_primary_bbs": {
                str(k): list(v) for k, v in self.duplicate_primary_bbs.items()
            },
            "uncovered_edges": [[s, d] for s, d in self.uncovered_edges],
            "invalid_bb_refs": list(self.invalid_bb_refs),
            "edge_explanation_counts": dict(self.edge_explanation_counts),
            "edge_explanation_preview": {
                str(k): list(v) for k, v in self.edge_explanation_preview.items()
            },
            "strict_edge_preservation_ok": bool(self.strict_edge_preservation_ok),
            "fallback_region_count": int(self.fallback_region_count),
            "fallback_bb_count": int(self.fallback_bb_count),
            "goto_count": int(self.goto_count),
            "break_count": int(self.break_count),
            "continue_count": int(self.continue_count),
            "early_exit_count": int(self.early_exit_count),
            "cleanup_exit_count": int(self.cleanup_exit_count),
            "escape_count": int(self.escape_count),
            "structured_control_count": int(self.structured_control_count),
            "structured_bb_count": int(self.structured_bb_count),
            "slot_count": int(self.slot_count),
            "slot_locality_score": float(self.slot_locality_score),
            "empty_control_count": int(self.empty_control_count),
            "empty_loop_count": int(self.empty_loop_count),
            "empty_if_count": int(self.empty_if_count),
            "empty_switch_count": int(self.empty_switch_count),
            "empty_control_with_context_count": int(self.empty_control_with_context_count),
            "control_body_total_bb_count": int(self.control_body_total_bb_count),
            "control_body_nested_bb_count": int(self.control_body_nested_bb_count),
            "body_bb_outside_control_count": int(self.body_bb_outside_control_count),
            "score": float(self.score),
        }


def _uniq_ints(xs: Iterable[int]) -> List[int]:
    return sorted({int(x) for x in xs})


def _fmt_bbs(xs: Sequence[int], *, max_items: int = 18) -> str:
    vals = _uniq_ints(xs)
    if len(vals) > max_items:
        return "[" + ", ".join(str(x) for x in vals[:max_items]) + ", ...]"
    return "[" + ", ".join(str(x) for x in vals) + "]"


def _fmt_edge(edge: Edge) -> str:
    return f"BB_{int(edge[0])}->BB_{int(edge[1])}"


def _escape_slot_types() -> Set[str]:
    return {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}


def _normalise_edges(raw_edge_index: Any, n_bb: int) -> List[Edge]:
    if not (isinstance(raw_edge_index, list) and len(raw_edge_index) == 2):
        return []
    src_raw, dst_raw = raw_edge_index
    if not isinstance(src_raw, list) or not isinstance(dst_raw, list):
        return []
    edges: Set[Edge] = set()
    for s, d in zip(src_raw, dst_raw):
        try:
            ss = int(s)
            dd = int(d)
        except Exception:
            continue
        if 0 <= ss < n_bb and 0 <= dd < n_bb:
            edges.add((ss, dd))
    return sorted(edges)


def _natural_loop_is_reducible(header: int, nodes: Set[int], preds: Dict[int, List[int]]) -> bool:
    header = int(header)
    node_set = {int(x) for x in nodes}
    if header not in node_set:
        return False
    for n in node_set:
        if n == header:
            continue
        if any(int(p) not in node_set for p in preds.get(n, [])):
            return False
    return True


def _compute_sese_regions(
    *,
    n_bb: int,
    edges: Sequence[Edge],
    succs: Dict[int, List[int]],
    preds: Dict[int, List[int]],
    dom: Sequence[Set[int]],
    pdom: Sequence[Set[int]],
    loop_nodes_by_header: Dict[int, Set[int]],
    max_region_nodes: int = 96,
) -> List[Dict[str, Any]]:
    """Compute conservative single-entry/single-exit regions.

    This is intentionally stricter than a full interval structurer.  It only
    emits regions that can be validated against the original CFG without
    duplicating blocks or inventing control flow.
    """
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, Tuple[int, ...]]] = set()
    loop_sets = [set(int(x) for x in xs) for xs in loop_nodes_by_header.values() if xs]

    for entry in range(n_bb):
        if entry >= len(pdom):
            continue
        exit_candidates = sorted(int(x) for x in (set(pdom[entry]) - {entry}) if 0 <= int(x) < n_bb)
        for exit_bb in exit_candidates:
            region = {
                n
                for n in range(n_bb)
                if entry in dom[n] and exit_bb in pdom[n] and n != exit_bb
            }
            if entry not in region:
                continue
            if len(region) < 2 or len(region) > int(max_region_nodes):
                continue
            if any((int(s) in region and int(d) == entry) for s, d in edges):
                continue

            incoming = [
                (int(s), int(d))
                for s, d in edges
                if int(d) in region and int(s) not in region
            ]
            outgoing = [
                (int(s), int(d))
                for s, d in edges
                if int(s) in region and int(d) not in region
            ]
            if any(d != entry for _s, d in incoming):
                continue
            exits = sorted({d for _s, d in outgoing})
            if exits and exits != [exit_bb]:
                continue
            if exit_bb in region:
                continue
            if not any(int(d) == exit_bb for _s, d in outgoing):
                # No real exit edge means this is usually a prefix artifact, not
                # a useful structuring interval.
                continue

            crosses_loop = False
            for loop in loop_sets:
                overlap = region & loop
                if overlap and overlap != loop:
                    crosses_loop = True
                    break
            if crosses_loop:
                continue

            key = (entry, exit_bb, tuple(sorted(region)))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "entry": int(entry),
                    "exit": int(exit_bb),
                    "nodes": sorted(int(x) for x in region),
                    "incoming_edges": [[int(s), int(d)] for s, d in incoming],
                    "outgoing_edges": [[int(s), int(d)] for s, d in outgoing],
                }
            )

    # Prefer regions that can cover more BBs, but keep deterministic order.
    out.sort(key=lambda r: (-len(r["nodes"]), int(r["entry"]), int(r["exit"])))
    pruned: List[Dict[str, Any]] = []
    region_sets: List[Set[int]] = []
    for reg in out:
        nodes = set(int(x) for x in reg.get("nodes") or [])
        duplicate = any(nodes == old for old in region_sets)
        if duplicate:
            continue
        pruned.append(reg)
        region_sets.append(nodes)
    return pruned


def _compute_sccs_from_succs(succs: Dict[int, List[int]], n_bb: int) -> List[Set[int]]:
    """Tarjan SCCs over normalized BB ids."""
    index = 0
    stack: List[int] = []
    on_stack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    sccs: List[Set[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in succs.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: Set[int] = set()
            while stack:
                w = stack.pop()
                on_stack.remove(w)
                comp.add(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n_bb):
        if v not in indices:
            strongconnect(v)
    return sorted(sccs, key=lambda xs: (min(xs) if xs else -1, len(xs)))


def _succ_pred(edges: Sequence[Edge], n_bb: int) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    succs: Dict[int, List[int]] = {i: [] for i in range(n_bb)}
    preds: Dict[int, List[int]] = {i: [] for i in range(n_bb)}
    for s, d in edges:
        succs[int(s)].append(int(d))
        preds[int(d)].append(int(s))
    for i in range(n_bb):
        succs[i] = sorted(set(succs[i]))
        preds[i] = sorted(set(preds[i]))
    return succs, preds


def _dfs_order(succs: Dict[int, List[int]], n_bb: int) -> Tuple[List[int], List[int]]:
    visited: Set[int] = set()
    order: List[int] = []
    stack: List[int] = [0] if n_bb > 0 else []
    while stack:
        node = int(stack.pop())
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for succ in reversed(succs.get(node, [])):
            if succ not in visited:
                stack.append(succ)
    unreachable = [i for i in range(n_bb) if i not in visited]
    order.extend(unreachable)
    return order, unreachable


def _reachable_from_entry(succs: Dict[int, List[int]], n_bb: int, entry: int = 0) -> Set[int]:
    seen: Set[int] = set()
    stack: List[int] = [entry] if 0 <= entry < n_bb else []
    while stack:
        n = int(stack.pop())
        if n in seen:
            continue
        seen.add(n)
        stack.extend(s for s in succs.get(n, []) if s not in seen)
    return seen


def _dominators(succs: Dict[int, List[int]], preds: Dict[int, List[int]], n_bb: int) -> List[Set[int]]:
    if n_bb <= 0:
        return []
    reachable = _reachable_from_entry(succs, n_bb, 0)
    all_reach = set(reachable)
    dom: List[Set[int]] = []
    for i in range(n_bb):
        if i == 0:
            dom.append({0})
        elif i in reachable:
            dom.append(set(all_reach))
        else:
            dom.append({i})
    changed = True
    while changed:
        changed = False
        for n in sorted(reachable - {0}):
            pred_reach = [p for p in preds.get(n, []) if p in reachable]
            if not pred_reach:
                new = {n}
            else:
                inter = set(dom[pred_reach[0]])
                for p in pred_reach[1:]:
                    inter &= dom[p]
                new = inter | {n}
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


def _immediate_from_dom(dom: Sequence[Set[int]], n_bb: int, entry: int) -> List[int]:
    out = [-1] * n_bb
    if 0 <= entry < n_bb:
        out[entry] = entry
    for n in range(n_bb):
        if n == entry:
            continue
        strict = set(dom[n]) - {n}
        if not strict:
            continue
        # Immediate dominator is the strict dominator not dominated by another
        # strict dominator descendant in this set.
        best = None
        for cand in strict:
            dominated_by_other = any(cand in dom[other] for other in strict if other != cand)
            if not dominated_by_other:
                best = cand
                break
        out[n] = int(best) if best is not None else int(max(strict))
    return out


def _postdominators(succs: Dict[int, List[int]], preds: Dict[int, List[int]], n_bb: int) -> List[Set[int]]:
    if n_bb <= 0:
        return []
    exits = [i for i in range(n_bb) if not succs.get(i)]
    if not exits:
        # Cyclic no-exit CFG: make every node only postdominate itself.  The MVP
        # then falls back to explicit gotos instead of over-structuring.
        return [{i} for i in range(n_bb)]
    all_nodes = set(range(n_bb))
    pdom: List[Set[int]] = []
    for i in range(n_bb):
        pdom.append({i} if i in exits else set(all_nodes))
    changed = True
    while changed:
        changed = False
        for n in range(n_bb):
            if n in exits:
                continue
            succ_list = succs.get(n, [])
            if not succ_list:
                new = {n}
            else:
                inter = set(pdom[succ_list[0]])
                for s in succ_list[1:]:
                    inter &= pdom[s]
                new = inter | {n}
            if new != pdom[n]:
                pdom[n] = new
                changed = True
    return pdom


def _immediate_postdom(pdom: Sequence[Set[int]], n_bb: int) -> List[int]:
    out = [-1] * n_bb
    for n in range(n_bb):
        strict = set(pdom[n]) - {n}
        if not strict:
            continue
        best = None
        for cand in strict:
            dominated_by_other = any(cand in pdom[other] for other in strict if other != cand)
            if not dominated_by_other:
                best = cand
                break
        out[n] = int(best) if best is not None else int(min(strict))
    return out


def _natural_loop_nodes(header: int, back_src: int, preds: Dict[int, List[int]]) -> Set[int]:
    nodes: Set[int] = {int(header)}
    stack: List[int] = [int(back_src)]
    while stack:
        n = int(stack.pop())
        if n in nodes:
            continue
        nodes.add(n)
        for p in preds.get(n, []):
            if p not in nodes:
                stack.append(p)
    return nodes


def analyse_cfg(row: Dict[str, Any]) -> CFGView:
    bl = row.get("bin_block_lengths") or []
    n_bb = len(bl) if isinstance(bl, list) else 0
    edges = _normalise_edges(row.get("bin_edge_index"), n_bb)
    succs, preds = _succ_pred(edges, n_bb)
    dfs_order, unreachable = _dfs_order(succs, n_bb)
    dom = _dominators(succs, preds, n_bb)
    idom = _immediate_from_dom(dom, n_bb, 0)
    pdom = _postdominators(succs, preds, n_bb)
    ipdom = _immediate_postdom(pdom, n_bb)

    back_edges: List[Edge] = []
    loop_nodes_by_header: Dict[int, Set[int]] = {}
    for u, v in edges:
        if 0 <= v < len(dom) and v in dom[u] and u != v:
            back_edges.append((u, v))
            loop_nodes_by_header.setdefault(v, set()).update(_natural_loop_nodes(v, u, preds))
    sccs = _compute_sccs_from_succs(succs, n_bb)
    reducible_loops = [
        set(int(x) for x in nodes)
        for h, nodes in loop_nodes_by_header.items()
        if _natural_loop_is_reducible(int(h), set(nodes), preds)
    ]
    irreducible_sccs: List[Set[int]] = []
    for comp in sccs:
        if len(comp) <= 1:
            continue
        if not any(comp <= loop for loop in reducible_loops):
            irreducible_sccs.append(set(comp))
    sese_regions = _compute_sese_regions(
        n_bb=n_bb,
        edges=edges,
        succs=succs,
        preds=preds,
        dom=dom,
        pdom=pdom,
        loop_nodes_by_header=loop_nodes_by_header,
    )

    return CFGView(
        n_bb=n_bb,
        edges=edges,
        succs=succs,
        preds=preds,
        dfs_order=dfs_order,
        unreachable=unreachable,
        dom=dom,
        idom=idom,
        pdom=pdom,
        ipdom=ipdom,
        sccs=sccs,
        irreducible_sccs=irreducible_sccs,
        back_edges=sorted(back_edges),
        loop_nodes_by_header=loop_nodes_by_header,
        sese_regions=sese_regions,
    )


class _Builder:
    def __init__(self, cfg: CFGView) -> None:
        self.cfg = cfg
        self.slots: List[Slot] = []
        self.explicit_edges: Set[Edge] = set()
        self.structured_regions: List[Set[int]] = []
        self.fallback_regions: List[Set[int]] = []

    def slot(
        self,
        slot_type: str,
        primary: Sequence[int],
        context: Optional[Sequence[int]] = None,
        *,
        role: str = "",
        owner: Optional[str] = None,
        edge: Optional[Edge] = None,
        reason: str = "",
    ) -> Slot:
        sid = len(self.slots)
        s = Slot(
            slot_id=sid,
            slot_type=slot_type,
            role=role,
            primary_bbs=_uniq_ints(primary),
            context_bbs=_uniq_ints(context if context is not None else primary),
            owner=owner,
            edge=edge,
            reason=reason,
        )
        self.slots.append(s)
        if edge is not None:
            self.explicit_edges.add((int(edge[0]), int(edge[1])))
        return s

    def block(self, bbs: Sequence[int], *, role: str = "stmt") -> Node:
        primary = _uniq_ints(bbs)
        slot = self.slot("BLOCK", primary, primary, role=role)
        return Node(kind="BLOCK_SLOT", slot=slot, region_bbs=primary)

    def case_block(self, bbs: Sequence[int], *, role: str = "switch_case") -> Node:
        primary = _uniq_ints(bbs)
        slot = self.slot("CASE", primary, primary, role=role, reason="explicit_switch_case_slot")
        return Node(kind="CASE_SLOT", slot=slot, region_bbs=primary)

    def unstructured(self, bbs: Sequence[int], *, reason: str) -> Node:
        primary = _uniq_ints(bbs)
        slot = self.slot("UNSTRUCTURED_REGION", primary, primary, role="fallback", reason=reason)
        reg = set(primary)
        if reg:
            self.fallback_regions.append(reg)
        return Node(kind="UNSTRUCTURED_REGION", slot=slot, region_bbs=primary, reason=reason)

    def goto(self, edge: Edge, *, reason: str) -> Node:
        s, d = int(edge[0]), int(edge[1])
        slot = self.slot("GOTO", [], [s, d], role="goto", edge=(s, d), reason=reason)
        return Node(kind="GOTO_SLOT", slot=slot, region_bbs=[s, d], reason=reason)

    def break_slot(self, edge: Edge, *, reason: str) -> Node:
        s, d = int(edge[0]), int(edge[1])
        slot = self.slot("BREAK", [], [s, d], role="break", edge=(s, d), reason=reason)
        return Node(kind="BREAK_SLOT", slot=slot, region_bbs=[s, d], reason=reason)

    def continue_slot(self, edge: Edge, *, reason: str) -> Node:
        s, d = int(edge[0]), int(edge[1])
        slot = self.slot("CONTINUE", [], [s, d], role="continue", edge=(s, d), reason=reason)
        return Node(kind="CONTINUE_SLOT", slot=slot, region_bbs=[s, d], reason=reason)

    def early_exit_slot(self, edge: Edge, *, reason: str) -> Node:
        s, d = int(edge[0]), int(edge[1])
        slot = self.slot("EARLY_EXIT", [], [s, d], role="early_exit", edge=(s, d), reason=reason)
        return Node(kind="EARLY_EXIT_SLOT", slot=slot, region_bbs=[s, d], reason=reason)

    def cleanup_exit_slot(
        self,
        target: int,
        source_bbs: Sequence[int],
        *,
        reason: str,
    ) -> Node:
        target = int(target)
        sources = _uniq_ints(source_bbs)
        slot = self.slot(
            "CLEANUP_EXIT",
            [],
            sources + [target],
            role="cleanup_exit",
            owner=f"cleanup_exit_bb_{target}",
            reason=reason,
        )
        return Node(kind="CLEANUP_EXIT_SLOT", slot=slot, region_bbs=sources + [target], reason=reason)


def _collect_until_merge(
    cfg: CFGView,
    roots: Sequence[int],
    merge: int,
    stop: Set[int],
    *,
    max_nodes: int,
) -> Set[int]:
    nodes: Set[int] = set()
    stack: List[int] = [int(x) for x in roots]
    while stack:
        n = int(stack.pop())
        if n == merge or n in stop or n in nodes:
            continue
        nodes.add(n)
        if len(nodes) > max_nodes:
            return nodes
        for s in cfg.succs.get(n, []):
            if s != merge and s not in stop:
                stack.append(s)
    return nodes


def _loop_escape_node(
    builder: _Builder,
    edge: Edge,
    *,
    loop_header: int,
    loop_nodes: Set[int],
    reason: str,
) -> Node:
    """Classify loop escape edges without dropping CFG provenance.

    Edges still remain explicit CFG-preserving slots; this only distinguishes
    loop exits/continues from opaque goto debt for readability metrics.
    """
    s, d = int(edge[0]), int(edge[1])
    if d == int(loop_header):
        return builder.continue_slot((s, d), reason=reason)
    if s in loop_nodes and d not in loop_nodes:
        return builder.break_slot((s, d), reason=reason)
    return builder.goto((s, d), reason=reason)


def _classify_escape_node(
    builder: _Builder,
    edge: Edge,
    *,
    reason: str,
    loop_header: Optional[int] = None,
    loop_nodes: Optional[Set[int]] = None,
    early_exit_targets: Optional[Set[int]] = None,
) -> Node:
    """Choose the most precise explicit escape slot for a residual CFG edge."""
    s, d = int(edge[0]), int(edge[1])
    if loop_header is not None and loop_nodes is not None:
        return _loop_escape_node(
            builder,
            (s, d),
            loop_header=int(loop_header),
            loop_nodes=set(loop_nodes),
            reason=reason,
        )
    if early_exit_targets and d in {int(x) for x in early_exit_targets}:
        return builder.early_exit_slot((s, d), reason=reason)
    return builder.goto((s, d), reason=reason)


def _function_exit_like_nodes(cfg: CFGView) -> Set[int]:
    """Return nodes that are plausible canonical return/error exits.

    This is CFG-only: exit blocks and high-fan-in jump-to-exit blocks are treated
    as early-exit targets.  It improves readability without claiming source AST
    identity.
    """
    exits = {int(bb) for bb in range(cfg.n_bb) if not cfg.succs.get(int(bb))}
    out = set(exits)
    for bb in range(cfg.n_bb):
        succs = [int(x) for x in cfg.succs.get(int(bb), [])]
        if len(succs) == 1 and succs[0] in exits and len(cfg.preds.get(int(bb), [])) >= 2:
            out.add(int(bb))
    for bb in range(cfg.n_bb):
        if len(cfg.preds.get(int(bb), [])) >= 3 and (
            int(bb) in exits or any(int(s) in exits for s in cfg.succs.get(int(bb), []))
        ):
            out.add(int(bb))
    return out


def _branch_has_exit_like_path(cfg: CFGView, nodes: Set[int], exit_like: Set[int]) -> bool:
    node_set = {int(x) for x in nodes}
    exits = {int(x) for x in exit_like}
    if node_set & exits:
        return True
    for bb in node_set:
        succs = [int(x) for x in cfg.succs.get(int(bb), [])]
        if not succs or any(int(s) in exits for s in succs):
            return True
    return False


def _empty_exit_branch_target(
    cfg: CFGView,
    succs: Sequence[int],
    branch_sets: Sequence[Set[int]],
    selected_idx: Optional[int],
    merge: int,
    exit_like: Set[int],
) -> Optional[int]:
    """Return the exit target for an empty selected branch, if it is a guard.

    Optimized binaries often collapse `if (...) return;` into an edge from the
    condition BB directly to the postdominating exit BB.  A normal
    collect-until-merge pass sees that branch as empty because the successor is
    the merge.  Treating that edge as an explicit early-exit child gives the
    candidate pool a source-style guard shape without hiding the CFG edge.
    """
    if selected_idx is None:
        return None
    idx = int(selected_idx)
    if idx < 0 or idx >= len(succs) or idx >= len(branch_sets):
        return None
    if branch_sets[idx]:
        return None
    target = int(succs[idx])
    if target == int(merge) and target in {int(x) for x in exit_like}:
        return target
    if target in {int(x) for x in exit_like} and not cfg.succs.get(target):
        return target
    return None


def _preferred_guard_branch_index(
    cfg: CFGView,
    succs: Sequence[int],
    branch_sets: Sequence[Set[int]],
    merge: int,
    exit_like: Set[int],
) -> Optional[int]:
    """Pick the branch most likely to be a source-level guard/early exit."""
    exit_like = {int(x) for x in exit_like}
    empty_exit: List[int] = []
    for i, root in enumerate(succs):
        if _empty_exit_branch_target(cfg, succs, branch_sets, i, merge, exit_like) is not None:
            empty_exit.append(int(i))
    if empty_exit:
        return int(empty_exit[0])

    scored: List[Tuple[int, int, int]] = []
    for i, nodes_raw in enumerate(branch_sets):
        nodes = {int(x) for x in nodes_raw}
        if not nodes:
            continue
        root = int(succs[i]) if i < len(succs) else -1
        exitish = root in exit_like or _branch_has_exit_like_path(cfg, nodes, exit_like)
        if not exitish:
            continue
        other_sizes = [len(set(x)) for j, x in enumerate(branch_sets) if int(j) != int(i)]
        other_max = max(other_sizes, default=0)
        # Prefer short exit/error branches over the larger fallthrough body.
        if len(nodes) <= max(2, other_max):
            scored.append((len(nodes), int(i), root))
    if scored:
        scored.sort()
        return int(scored[0][1])
    return None


def _collect_guard_branch_sets_no_merge(
    cfg: CFGView,
    header: int,
    succs: Sequence[int],
    *,
    max_region_nodes: int,
) -> Tuple[List[Set[int]], bool]:
    branch_sets: List[Set[int]] = []
    too_large = False
    for s in succs:
        nodes = _collect_until_merge(
            cfg,
            [int(s)],
            -1,
            {int(header)},
            max_nodes=max_region_nodes,
        )
        if len(nodes) > int(max_region_nodes):
            too_large = True
        branch_sets.append(nodes)
    return branch_sets, bool(too_large)


def _if_policy_selected_index(policy: str, branch_count: int) -> Optional[int]:
    p = str(policy or "auto")
    if p in {"plain", "no_if", "suppress", "block_only"}:
        return None
    if p == "then_only":
        return 0 if int(branch_count) > 0 else None
    if p == "else_only":
        return 1 if int(branch_count) > 1 else None
    if p.startswith("only_"):
        try:
            idx = int(p.split("_", 1)[1])
        except Exception:
            return None
        return idx if 0 <= idx < int(branch_count) else None
    if p.startswith("branch_") and p.endswith("_only"):
        try:
            idx = int(p[len("branch_") : -len("_only")])
        except Exception:
            return None
        return idx if 0 <= idx < int(branch_count) else None
    return None


def _if_policy_suppresses_header(policy: str) -> bool:
    return str(policy or "") in {"plain", "no_if", "suppress", "block_only"}


def _linear_candidate(cfg: CFGView, *, name: str = "linear_goto") -> Candidate:
    b = _Builder(cfg)
    children: List[Node] = []
    order = list(cfg.dfs_order)
    pos = {bb: i for i, bb in enumerate(order)}
    for bb in order:
        children.append(b.block([bb], role="bb"))
        for dst in cfg.succs.get(bb, []):
            if pos.get(dst) == pos.get(bb, -10) + 1:
                b.explicit_edges.add((int(bb), int(dst)))
            else:
                children.append(b.goto((bb, dst), reason="non_linear_cfg_edge"))
    root = Node(kind="FUNCTION", children=children, region_bbs=list(range(cfg.n_bb)))
    return Candidate(
        name=name,
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
    )


def _full_fallback_candidate(cfg: CFGView) -> Candidate:
    b = _Builder(cfg)
    root = Node(
        kind="FUNCTION",
        children=[b.unstructured(list(range(cfg.n_bb)), reason="full_function_fallback")],
        region_bbs=list(range(cfg.n_bb)),
    )
    return Candidate(
        name="full_fallback",
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
        notes=["all CFG edges are preserved inside the fallback region"],
    )


def _order_helpers(cfg: CFGView) -> Tuple[Dict[int, int], Any]:
    order = list(cfg.dfs_order)
    order_pos = {int(bb): i for i, bb in enumerate(order)}

    def order_key(bb: int) -> Tuple[int, int]:
        return (int(order_pos.get(int(bb), int(bb))), int(bb))

    return order_pos, order_key


def _switch_chain_region(
    cfg: CFGView,
    start: int,
    *,
    max_region_nodes: int,
    conservative: bool = True,
) -> Optional[Tuple[List[int], Set[int], int]]:
    """Detect a switch lowered as a chain of binary tests.

    Optimized switch statements often no longer have a CFG node with out-degree
    >= 3.  This candidate family treats a forward chain of branch headers as one
    switch dispatch while keeping validator/goto fallback as the safety boundary.
    """
    start = int(start)
    if len(cfg.succs.get(start, [])) < 2:
        return None
    order_pos, order_key = _order_helpers(cfg)
    chain: List[int] = []
    case_roots: List[int] = []
    seen: Set[int] = set()
    cur = start
    max_chain = max(1, min(16, int(max_region_nodes)))
    while 0 <= cur < cfg.n_bb and cur not in seen and len(chain) < max_chain:
        succs = [int(s) for s in cfg.succs.get(cur, []) if 0 <= int(s) < cfg.n_bb]
        if len(succs) < 2:
            break
        seen.add(cur)
        chain.append(cur)
        next_headers = [
            s for s in succs
            if s not in seen
            and len(cfg.succs.get(s, [])) >= 2
            and order_pos.get(s, cfg.n_bb + s) > order_pos.get(cur, -1)
            and (cur, s) not in cfg.back_edges
        ]
        if not next_headers:
            case_roots.extend(succs)
            break
        nxt = min(next_headers, key=order_key)
        case_roots.extend(s for s in succs if s != nxt)
        cur = nxt

    chain_set = set(chain)
    case_roots = [r for r in _uniq_ints(case_roots) if r not in chain_set]
    if len(chain) < 2:
        return None
    if len(case_roots) < 3:
        return None

    merge = cfg.ipdom[start] if 0 <= start < len(cfg.ipdom) else -1
    merge_fan_in = len(cfg.preds.get(int(merge), [])) if 0 <= int(merge) < cfg.n_bb else 0
    stop = set(chain_set)
    case_body: Set[int] = set()
    for root in case_roots:
        if root == merge:
            continue
        nodes = _collect_until_merge(
            cfg,
            [root],
            merge,
            stop,
            max_nodes=max_region_nodes,
        )
        case_body.update(int(x) for x in nodes if 0 <= int(x) < cfg.n_bb)
    region = set(chain_set) | case_body
    if len(region) <= len(chain_set) or len(region) > int(max_region_nodes):
        return None
    if conservative:
        # Binary-test chains are easy to over-read as source switches.  Keep this
        # candidate family as a rare rescue path: a credible chain must have a
        # real join/default shape and must not swallow an ordinary loop body.
        loop_nodes = set().union(*cfg.loop_nodes_by_header.values()) if cfg.loop_nodes_by_header else set()
        if region & loop_nodes:
            return None
        if merge < 0 or merge in chain_set:
            return None
        if len(chain) < 3:
            return None
        if merge_fan_in < 3:
            return None
    else:
        loop_nodes = set().union(*cfg.loop_nodes_by_header.values()) if cfg.loop_nodes_by_header else set()
        if region & loop_nodes:
            return None
        if merge < 0 or merge in chain_set:
            return None
        # The relaxed switch-chain candidate is still a binary-only candidate,
        # not a source oracle.  Require a genuine compare chain with several
        # case/default roots plus a high-fan-in join as target-set evidence.
        if merge_fan_in < 3:
            return None
        if len(chain) < 3:
            return None
    return chain, region, int(merge)


def _switch_chain_candidate(
    cfg: CFGView,
    *,
    start: int,
    max_region_nodes: int = 64,
    conservative: bool = True,
    name: str = "cfg_switch_chain",
) -> Optional[Candidate]:
    detected = _switch_chain_region(
        cfg,
        int(start),
        max_region_nodes=max_region_nodes,
        conservative=bool(conservative),
    )
    if detected is None:
        return None
    chain, region, _merge = detected
    order_pos, order_key = _order_helpers(cfg)

    def ordered(nodes: Iterable[int]) -> List[int]:
        return sorted((int(x) for x in nodes), key=order_key)

    b = _Builder(cfg)
    children: List[Node] = []
    emitted_switch = False
    chain_set = set(chain)
    body_nodes = set(region) - chain_set
    cond = b.slot(
        "COND",
        ordered(chain_set),
        ordered(region),
        role="switch_chain_cond",
        owner=f"switch_chain_bb_{int(start)}",
    )
    switch_node = Node(
        kind="SWITCH",
        slot=cond,
        children=[b.case_block([bb], role="switch_case") for bb in ordered(body_nodes)],
        region_bbs=ordered(region),
        reason="binary_branch_chain_as_switch",
    )
    b.structured_regions.append(set(region))
    for edge in cfg.edges:
        s, d = int(edge[0]), int(edge[1])
        if s in region:
            # Dispatch-chain and case-exit edges are part of this switch
            # candidate's explicit lowering contract.
            b.explicit_edges.add((s, d))

    for bb in sorted(range(cfg.n_bb), key=order_key):
        if bb in region:
            if not emitted_switch and bb == int(start):
                children.append(switch_node)
                emitted_switch = True
            continue
        children.append(b.block([bb], role="bb"))
    if not emitted_switch:
        children.insert(0, switch_node)
    root = Node(kind="FUNCTION", children=children, region_bbs=list(range(cfg.n_bb)))
    cand = Candidate(
        name=name,
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
        notes=[
            "switch candidate from binary branch chain",
            f"switch_chain_conservative={bool(conservative)}",
        ],
    )
    _add_missing_edge_gotos(cfg, cand)
    return cand


def _switch_fanin_region(
    cfg: CFGView,
    start: int,
    *,
    max_region_nodes: int,
) -> Optional[Tuple[Set[int], int]]:
    """Detect jump-table / lowered-switch fan-in shape from CFG only.

    Common optimized switches have a range-check header and several case blocks
    that all jump to a shared join/return block.  The dispatch fanout may not be
    represented as direct CFG edges from the header after binary recovery, so the
    old out-degree>=3 or compare-chain detector misses it.  This detector only
    accepts compact regions with a high-fan-in join and at least three case-like
    predecessors, keeping it conservative and auditable.
    """
    start = int(start)
    if not (0 <= start < cfg.n_bb):
        return None
    order_pos, order_key = _order_helpers(cfg)
    start_pos = int(order_pos.get(start, start))
    succs = [int(s) for s in cfg.succs.get(start, []) if 0 <= int(s) < cfg.n_bb]
    if not succs:
        return None

    join_candidates = [
        int(bb)
        for bb in range(cfg.n_bb)
        if int(bb) != start
        and len(cfg.preds.get(int(bb), [])) >= 3
        and int(order_pos.get(int(bb), int(bb))) > start_pos
    ]
    if not join_candidates:
        return None

    best_score: Optional[Tuple[int, int, int, int]] = None
    best_region: Optional[Set[int]] = None
    best_join: int = -1
    for join in sorted(join_candidates, key=order_key):
        case_nodes = {
            int(p)
            for p in cfg.preds.get(int(join), [])
            if int(p) != start
            and 0 <= int(p) < cfg.n_bb
            and int(order_pos.get(int(p), int(p))) >= start_pos
        }
        # Include direct non-join successors from the range-check / dispatch BB.
        case_nodes.update(int(s) for s in succs if int(s) != join)
        case_nodes = {bb for bb in case_nodes if bb != start and bb != join}
        if len(case_nodes) < 3:
            continue
        region = {start} | case_nodes
        if len(region) < 4 or len(region) > int(max_region_nodes):
            continue
        if any(bb in cfg.loop_nodes_by_header for bb in region):
            continue
        if any((int(s), int(d)) in cfg.back_edges for s, d in cfg.edges if int(s) in region):
            continue
        incoming = [
            (int(s), int(d))
            for s, d in cfg.edges
            if int(d) in region and int(s) not in region
        ]
        if incoming:
            continue
        # Case blocks should be simple exits to the join or true exits.  Allow a
        # small amount of noise, but reject broad arbitrary subgraphs.
        simple_cases = 0
        for bb in case_nodes:
            outs = [int(x) for x in cfg.succs.get(int(bb), [])]
            if not outs or all(dst == join or not cfg.succs.get(dst) for dst in outs):
                simple_cases += 1
        if simple_cases < max(3, len(case_nodes) - 1):
            continue
        score = (len(case_nodes), simple_cases, -len(region), -join)
        if best_score is None or score > best_score:
            best_score = score
            best_region = set(region)
            best_join = int(join)
    if best_region is None:
        return None
    return set(best_region), int(best_join)


def _switch_fanin_candidate(
    cfg: CFGView,
    *,
    start: int,
    max_region_nodes: int = 64,
    name: str = "cfg_switch_fanin",
) -> Optional[Candidate]:
    detected = _switch_fanin_region(cfg, int(start), max_region_nodes=max_region_nodes)
    if detected is None:
        return None
    region, join = detected
    order_pos, order_key = _order_helpers(cfg)

    def ordered(nodes: Iterable[int]) -> List[int]:
        return sorted((int(x) for x in nodes), key=order_key)

    b = _Builder(cfg)
    children: List[Node] = []
    emitted_switch = False
    body_nodes = set(region) - {int(start)}
    cond = b.slot(
        "COND",
        [int(start)],
        ordered(region),
        role="switch_cond_fanin",
        owner=f"switch_fanin_bb_{int(start)}",
    )
    switch_node = Node(
        kind="SWITCH",
        slot=cond,
        children=[b.case_block([bb], role="switch_case") for bb in ordered(body_nodes)],
        region_bbs=ordered(region),
        reason=f"binary_fanin_switch_join_{int(join)}",
    )
    b.structured_regions.append(set(region))
    for s, d in cfg.edges:
        s, d = int(s), int(d)
        if s in region and (d in region or d == int(join) or not cfg.succs.get(d)):
            b.explicit_edges.add((s, d))

    for bb in sorted(range(cfg.n_bb), key=order_key):
        if bb in region:
            if not emitted_switch and bb == int(start):
                children.append(switch_node)
                emitted_switch = True
            continue
        children.append(b.block([bb], role="bb"))
    if not emitted_switch:
        children.insert(0, switch_node)
    root = Node(kind="FUNCTION", children=children, region_bbs=list(range(cfg.n_bb)))
    cand = Candidate(
        name=name,
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
        notes=["switch candidate from high-fan-in case join"],
    )
    _add_missing_edge_gotos(cfg, cand)
    return cand


def _select_sese_regions(
    cfg: CFGView,
    *,
    max_region_nodes: int,
    max_regions: int = 8,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    assigned: Set[int] = set()
    irreducible = [
        set(int(x) for x in comp)
        for comp in getattr(cfg, "irreducible_sccs", [])
        if comp
    ]
    for reg in cfg.sese_regions:
        nodes = {int(x) for x in (reg.get("nodes") or [])}
        if len(nodes) < 2 or len(nodes) > int(max_region_nodes):
            continue
        if any(nodes & comp for comp in irreducible):
            continue
        if nodes & assigned:
            continue
        selected.append(reg)
        assigned.update(nodes)
        if len(selected) >= int(max_regions):
            break
    return selected


def _sese_interval_grounded_seq_children(
    cfg: CFGView,
    builder: _Builder,
    nodes: Set[int],
    order_key: Any,
) -> List[Node]:
    """Emit an interval sequence without hiding branch provenance.

    The pure interval candidate is useful because it covers a large structured
    region without fallback.  If it renders every BB as an opaque block, though,
    Module-B loses the explicit COND_SLOT -> branch/SSA grounding chain required
    by the v3 contract.  This helper keeps the interval layout but emits a
    lightweight control node for every branch/loop header, while preserving all
    original header outgoing CFG edges explicitly.
    """

    def ordered(xs: Iterable[int]) -> List[int]:
        return sorted((int(x) for x in xs), key=order_key)

    out: List[Node] = []
    region_nodes = {int(x) for x in nodes if 0 <= int(x) < cfg.n_bb}
    assigned: Set[int] = set()

    def fallback_or_blocks(body_nodes: Iterable[int], *, reason: str) -> List[Node]:
        body = ordered(int(x) for x in body_nodes if int(x) in region_nodes)
        if not body:
            return []
        # Keep C slots local to the owning control node.  If the interval helper
        # cannot safely split a nested source-like body, per-BB block slots are
        # still better for Module C than rendering those BBs after the control.
        return [builder.block([int(bb)], role=reason) for bb in body]

    def disjoint(xs: Sequence[Set[int]]) -> bool:
        seen: Set[int] = set()
        for item in xs:
            if seen & item:
                return False
            seen.update(item)
        return True

    def branch_sets_for_header(bb: int, succs: Sequence[int]) -> List[Set[int]]:
        merge = cfg.ipdom[int(bb)] if 0 <= int(bb) < len(cfg.ipdom) else -1
        if merge < 0 or merge == int(bb):
            return [{int(s)} for s in succs if int(s) in region_nodes]
        out: List[Set[int]] = []
        too_large = False
        for succ in succs:
            if int(succ) == int(merge):
                out.append(set())
                continue
            nodes = _collect_until_merge(
                cfg,
                [int(succ)],
                int(merge),
                {int(bb)},
                max_nodes=len(region_nodes),
            )
            nodes = {int(x) for x in nodes if int(x) in region_nodes}
            if len(nodes) > len(region_nodes):
                too_large = True
            out.append(nodes)
        if too_large or not disjoint(out):
            return [{int(s)} for s in succs if int(s) in region_nodes]
        return out

    for bb in ordered(region_nodes):
        if int(bb) in assigned:
            continue
        succs = [int(s) for s in cfg.succs.get(int(bb), []) if int(s) in region_nodes]
        loop_nodes = set(int(x) for x in cfg.loop_nodes_by_header.get(int(bb), set())) & region_nodes
        out_degree = len(cfg.succs.get(int(bb), []))
        # A natural-loop header can be an unconditional jump target after binary
        # normalization.  Do not emit a fillable condition slot unless the CFG has
        # an actual branch at that BB; otherwise slot grounding would falsely ask
        # Module C to recover a condition SSA that does not exist.
        is_loop_header = bool(loop_nodes) and out_degree >= 2
        is_branch_header = out_degree >= 2

        if is_loop_header or is_branch_header:
            for dst in cfg.succs.get(int(bb), []):
                builder.explicit_edges.add((int(bb), int(dst)))

            if is_loop_header:
                control_region = set(loop_nodes) | {int(bb)}
                body_nodes = set(control_region) - {int(bb)}
                cond = builder.slot(
                    "COND",
                    [int(bb)],
                    ordered(control_region),
                    role="loop_cond_sese_interval",
                    owner=f"loop_bb_{int(bb)}",
                )
                out.append(
                    Node(
                        kind="LOOP",
                        slot=cond,
                        children=fallback_or_blocks(body_nodes, reason="sese_interval_loop_body_bb"),
                        region_bbs=ordered(control_region),
                        reason="sese_interval_loop_header_grounding",
                    )
                )
                builder.structured_regions.append(set(control_region))
                assigned.update(control_region)
                continue

            control_region = {int(bb)} | set(succs)
            if len(cfg.succs.get(int(bb), [])) >= 3:
                branch_sets = branch_sets_for_header(int(bb), succs)
                body_nodes = set().union(*branch_sets) if branch_sets else set(control_region) - {int(bb)}
                control_region = {int(bb)} | set(body_nodes)
                cond = builder.slot(
                    "COND",
                    [int(bb)],
                    ordered(control_region),
                    role="switch_cond_sese_interval",
                    owner=f"switch_bb_{int(bb)}",
                )
                out.append(
                    Node(
                        kind="SWITCH",
                        slot=cond,
                        children=fallback_or_blocks(body_nodes, reason="sese_interval_switch_case_bb"),
                        region_bbs=ordered(control_region),
                        reason="sese_interval_switch_header_grounding",
                    )
                )
            else:
                branch_sets = branch_sets_for_header(int(bb), succs)
                then_nodes = set(branch_sets[0]) if branch_sets else set()
                else_nodes = set().union(*branch_sets[1:]) if len(branch_sets) > 1 else set()
                if not then_nodes and succs:
                    then_nodes = {int(succs[0])}
                control_region = {int(bb)} | then_nodes | else_nodes
                cond = builder.slot(
                    "COND",
                    [int(bb)],
                    ordered(control_region),
                    role="if_cond_sese_interval",
                    owner=f"if_bb_{int(bb)}",
                )
                out.append(
                    Node(
                        kind="IF_ELSE" if else_nodes else "IF",
                        slot=cond,
                        children=fallback_or_blocks(then_nodes, reason="sese_interval_if_then_bb"),
                        else_children=fallback_or_blocks(else_nodes, reason="sese_interval_if_else_bb"),
                        region_bbs=ordered(control_region),
                        reason="sese_interval_if_header_grounding",
                    )
                )
            builder.structured_regions.append(set(control_region))
            assigned.update(control_region)
            continue

        out.append(builder.block([int(bb)], role="sese_interval_bb"))
        assigned.add(int(bb))
    return out


def _sese_interval_sequence_candidate(
    cfg: CFGView,
    *,
    max_region_nodes: int = 64,
    name: str = "cfg_sese_interval_seq",
) -> Optional[Candidate]:
    selected = _select_sese_regions(cfg, max_region_nodes=max_region_nodes)
    if not selected:
        return None
    order_pos, order_key = _order_helpers(cfg)

    def ordered(nodes: Iterable[int]) -> List[int]:
        return sorted((int(x) for x in nodes), key=order_key)

    by_entry = {int(reg["entry"]): reg for reg in selected}
    b = _Builder(cfg)
    children: List[Node] = []
    assigned: Set[int] = set()
    for bb in sorted(range(cfg.n_bb), key=order_key):
        if bb in assigned:
            continue
        reg = by_entry.get(int(bb))
        if reg is None:
            children.append(b.block([bb], role="bb"))
            assigned.add(bb)
            continue
        nodes = {int(x) for x in (reg.get("nodes") or [])}
        seq_children = _sese_interval_grounded_seq_children(
            cfg,
            b,
            nodes,
            order_key,
        )
        children.append(
            Node(
                kind="SEQ",
                children=seq_children,
                region_bbs=ordered(nodes),
                reason=f"sese_interval_entry_{int(reg['entry'])}_exit_{int(reg['exit'])}",
            )
        )
        b.structured_regions.append(set(nodes))
        assigned.update(nodes)

    root = Node(kind="FUNCTION", children=children, region_bbs=list(range(cfg.n_bb)))
    cand = Candidate(
        name=name,
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
        interval_regions=[set(int(x) for x in reg.get("nodes") or []) for reg in selected],
        notes=["SEQ/interval candidate from conservative SESE regions"],
    )
    _add_missing_edge_gotos(cfg, cand)
    return cand


def _sese_augmented_recursive_candidate(
    cfg: CFGView,
    *,
    max_region_nodes: int = 64,
    name: str = "cfg_recursive_sese_interval",
) -> Optional[Candidate]:
    selected = _select_sese_regions(cfg, max_region_nodes=max_region_nodes)
    if not selected:
        return None
    cand = _structured_candidate(
        cfg,
        max_region_nodes=max_region_nodes,
        if_policy="auto",
        name=name,
    )
    existing = {tuple(sorted(int(x) for x in reg)) for reg in cand.structured_regions}
    added = 0
    for reg in selected:
        nodes = tuple(sorted(int(x) for x in (reg.get("nodes") or [])))
        if nodes and nodes not in existing:
            cand.structured_regions.append(set(nodes))
            cand.interval_regions.append(set(nodes))
            existing.add(nodes)
            added += 1
    cand.notes.append(f"added {added} conservative SESE/interval structured regions")
    return cand


def _shallow_structured_candidate(
    cfg: CFGView,
    *,
    max_region_nodes: int = 64,
    if_policy: str = "auto",
    if_policy_by_header: Optional[Dict[int, str]] = None,
    name: str = "cfg_shallow_mvp",
) -> Candidate:
    """Single-level conservative structurer used as a candidate family member.

    This is close to the initial MVP-0 behavior: recognize top-level simple
    loop/if regions, keep their bodies as fallback regions, and rely on the
    validator to add explicit gotos for residual CFG edges.  It is intentionally
    retained for oracle/ranking experiments because it often has fewer slots and
    less over-fragmentation than the recursive candidate.
    """
    b = _Builder(cfg)
    children: List[Node] = []
    assigned: Set[int] = set()
    order = list(cfg.dfs_order)
    exit_like_nodes = _function_exit_like_nodes(cfg)

    for bb in order:
        if bb in assigned:
            continue

        loop_nodes = set(cfg.loop_nodes_by_header.get(bb, set()))
        if loop_nodes and len(loop_nodes) <= max_region_nodes and not (loop_nodes & assigned):
            body_nodes = _uniq_ints(loop_nodes - {bb})
            cond = b.slot("COND", [bb], _uniq_ints(loop_nodes), role="loop_cond", owner=f"loop_bb_{bb}")
            body = [b.unstructured(body_nodes, reason="loop_body_mvp_fallback")] if body_nodes else []
            children.append(
                Node(
                    kind="LOOP",
                    slot=cond,
                    children=body,
                    region_bbs=_uniq_ints(loop_nodes),
                )
            )
            b.structured_regions.append(set(loop_nodes))
            assigned.update(loop_nodes)
            for s0, d0 in cfg.edges:
                if s0 in loop_nodes and d0 not in loop_nodes:
                    if s0 == bb:
                        b.explicit_edges.add((s0, d0))
                    else:
                        children.append(
                            _loop_escape_node(
                                b,
                                (s0, d0),
                                loop_header=bb,
                                loop_nodes=set(loop_nodes),
                                reason="loop_escape_edge",
                            )
                        )
            continue

        succ = cfg.succs.get(bb, [])
        merge = cfg.ipdom[bb] if 0 <= bb < len(cfg.ipdom) else -1
        if len(succ) >= 2 and (
            (merge >= 0 and merge != bb)
            or str((if_policy_by_header or {}).get(int(bb), if_policy)) == "guard_exit"
        ):
            branch_sets: List[Set[int]] = []
            too_large = False
            if merge >= 0 and merge != bb:
                for s in succ:
                    nodes = _collect_until_merge(
                        cfg,
                        [s],
                        merge,
                        {bb},
                        max_nodes=max_region_nodes,
                    )
                    if len(nodes) > max_region_nodes:
                        too_large = True
                    branch_sets.append(nodes)
            else:
                branch_sets, too_large = _collect_guard_branch_sets_no_merge(
                    cfg,
                    int(bb),
                    succ,
                    max_region_nodes=max_region_nodes,
                )
            union = set().union(*branch_sets) if branch_sets else set()
            seen_branch: Set[int] = set()
            pairwise_disjoint = True
            for nodes in branch_sets:
                if seen_branch & nodes:
                    pairwise_disjoint = False
                    break
                seen_branch.update(nodes)
            region = {bb} | union
            if (
                not too_large
                and union
                and pairwise_disjoint
                and not (region & assigned)
                and len(region) <= max_region_nodes
            ):
                local_policy = str((if_policy_by_header or {}).get(int(bb), if_policy))
                selected_idx: Optional[int] = None
                fallthrough_roots: Set[int] = set()
                if local_policy == "guard_exit":
                    selected_idx = _preferred_guard_branch_index(
                        cfg,
                        succ,
                        branch_sets,
                        merge,
                        exit_like_nodes,
                    )
                else:
                    selected_idx = _if_policy_selected_index(local_policy, len(branch_sets))

                if selected_idx is None:
                    then_set = set(branch_sets[0]) if branch_sets else set()
                    else_set = set().union(*branch_sets[1:]) if len(branch_sets) > 1 else set()
                    control_region = set(region)
                else:
                    then_set = set(branch_sets[selected_idx])
                    early_exit_target = _empty_exit_branch_target(
                        cfg,
                        succ,
                        branch_sets,
                        selected_idx,
                        merge,
                        exit_like_nodes,
                    )
                    if not then_set and early_exit_target is None:
                        selected_idx = None
                        then_set = set(branch_sets[0]) if branch_sets else set()
                        else_set = set().union(*branch_sets[1:]) if len(branch_sets) > 1 else set()
                        control_region = set(region)
                    else:
                        else_set = set()
                        control_region = {bb} | then_set
                        for i, root in enumerate(succ):
                            if i != selected_idx and root != merge:
                                fallthrough_roots.add(int(root))

                cond = b.slot(
                    "COND",
                    [bb],
                    _uniq_ints(control_region),
                    role=f"if_cond_{local_policy}",
                    owner=f"if_bb_{bb}",
                )
                then_nodes = _uniq_ints(then_set)
                else_nodes = _uniq_ints(else_set)
                then_child = [b.unstructured(then_nodes, reason="if_then_mvp_fallback")] if then_nodes else []
                if selected_idx is not None and not then_nodes:
                    early_exit_target = _empty_exit_branch_target(
                        cfg,
                        succ,
                        branch_sets,
                        selected_idx,
                        merge,
                        exit_like_nodes,
                    )
                    if early_exit_target is not None:
                        then_child = [
                            b.early_exit_slot(
                                (int(bb), int(early_exit_target)),
                                reason="guard_empty_branch_early_exit",
                            )
                        ]
                else_child = [b.unstructured(else_nodes, reason="if_else_mvp_fallback")] if else_nodes else []
                children.append(
                    Node(
                        kind="IF_ELSE" if else_child else "IF",
                        slot=cond,
                        children=then_child,
                        else_children=else_child,
                        region_bbs=_uniq_ints(control_region),
                    )
                )
                b.structured_regions.append(set(control_region))
                assigned.update(control_region)
                for s0, d0 in cfg.edges:
                    if s0 == bb and d0 in fallthrough_roots:
                        b.explicit_edges.add((s0, d0))
                    elif s0 in control_region and d0 not in control_region and d0 != merge:
                        children.append(b.goto((s0, d0), reason="if_escape_edge"))
                    elif s0 in control_region and d0 == merge:
                        b.explicit_edges.add((s0, d0))
                continue

        children.append(b.block([bb], role="bb"))
        assigned.add(bb)
        for dst in cfg.succs.get(bb, []):
            if dst not in assigned and dst != bb:
                continue
            if dst != bb:
                children.append(b.goto((bb, dst), reason="back_or_cross_edge"))

    root = Node(kind="FUNCTION", children=children, region_bbs=list(range(cfg.n_bb)))
    cand = Candidate(
        name=name,
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
    )
    _add_missing_edge_gotos(cfg, cand)
    return cand


def _structured_candidate(
    cfg: CFGView,
    *,
    max_region_nodes: int = 64,
    max_control_depth: int = 16,
    min_branch_region_nodes: int = 1,
    if_policy: str = "auto",
    if_policy_by_header: Optional[Dict[int, str]] = None,
    name: str = "cfg_recursive_mvp",
) -> Candidate:
    b = _Builder(cfg)
    order = list(cfg.dfs_order)
    order_pos = {int(bb): i for i, bb in enumerate(order)}

    def order_key(bb: int) -> Tuple[int, int]:
        return (int(order_pos.get(int(bb), int(bb))), int(bb))

    def ordered(nodes: Iterable[int]) -> List[int]:
        return sorted((int(x) for x in nodes), key=order_key)

    exit_like_nodes = _function_exit_like_nodes(cfg)

    def is_switch_header(bb: int, region_nodes: Set[int]) -> bool:
        succ = [s for s in cfg.succs.get(int(bb), []) if s in region_nodes]
        return len(succ) >= 3

    def branch_sets_for_header(
        bb: int,
        region_nodes: Set[int],
        merge: int,
    ) -> Tuple[List[Set[int]], bool]:
        branch_sets: List[Set[int]] = []
        too_large = False
        for succ in cfg.succs.get(int(bb), []):
            if succ == merge:
                branch_sets.append(set())
                continue
            if succ not in region_nodes:
                branch_sets.append(set())
                continue
            nodes = _collect_until_merge(
                cfg,
                [succ],
                merge,
                {int(bb)},
                max_nodes=max_region_nodes,
            )
            nodes = {int(x) for x in nodes if int(x) in region_nodes}
            if len(nodes) > max_region_nodes:
                too_large = True
            branch_sets.append(nodes)
        return branch_sets, too_large

    def disjoint_branch_sets(branch_sets: Sequence[Set[int]]) -> bool:
        seen: Set[int] = set()
        for nodes in branch_sets:
            if seen & nodes:
                return False
            seen.update(nodes)
        return True

    def emit_escape_edges(
        parent_children: List[Node],
        region: Set[int],
        *,
        merge: int = -1,
        reason: str,
        header: Optional[int] = None,
        header_false_is_fallthrough: bool = False,
        loop_header: Optional[int] = None,
    ) -> None:
        for s0, d0 in cfg.edges:
            if s0 not in region or d0 in region:
                continue
            if d0 == merge:
                b.explicit_edges.add((s0, d0))
                continue
            if header_false_is_fallthrough and header is not None and s0 == int(header):
                b.explicit_edges.add((s0, d0))
                continue
            if loop_header is not None:
                parent_children.append(
                    _loop_escape_node(
                        b,
                        (s0, d0),
                        loop_header=int(loop_header),
                        loop_nodes=set(region),
                        reason=reason,
                    )
                )
                continue
            parent_children.append(b.goto((s0, d0), reason=reason))

    def structure_plain_region(nodes: Set[int], *, role: str, reason: str) -> List[Node]:
        if not nodes:
            return []
        # A single ordinary BB is a normal Module-C block target, not an
        # unstructured region.  For multiple ordinary BBs, emit stable per-BB
        # blocks and let the validator add explicit gotos for non-linear edges.
        out: List[Node] = []
        for bb in ordered(nodes):
            out.append(b.block([bb], role=role))
        return out if out else [b.unstructured(ordered(nodes), reason=reason)]

    def structure_region(
        region_nodes_raw: Iterable[int],
        *,
        role: str,
        depth: int,
    ) -> List[Node]:
        region_nodes = {int(x) for x in region_nodes_raw if 0 <= int(x) < cfg.n_bb}
        if not region_nodes:
            return []
        if len(region_nodes) > max_region_nodes:
            return [b.unstructured(ordered(region_nodes), reason="recursive_limit_fallback")]

        children: List[Node] = []
        assigned: Set[int] = set()
        control_depth_exhausted = depth >= int(max_control_depth)

        for bb in ordered(region_nodes):
            if bb in assigned:
                continue

            # Natural loop: use the CFG back-edge region and recursively
            # structure the loop body instead of always falling back.
            loop_nodes = set(cfg.loop_nodes_by_header.get(bb, set())) & region_nodes
            if (
                not control_depth_exhausted
                and loop_nodes
                and len(loop_nodes) <= max_region_nodes
                and not (loop_nodes & assigned)
            ):
                body_nodes = set(loop_nodes) - {bb}
                cond = b.slot("COND", [bb], ordered(loop_nodes), role="loop_cond", owner=f"loop_bb_{bb}")
                body = structure_region(body_nodes, role="loop_body", depth=depth + 1)
                node = Node(
                    kind="LOOP",
                    slot=cond,
                    children=body,
                    region_bbs=ordered(loop_nodes),
                )
                children.append(node)
                b.structured_regions.append(set(loop_nodes))
                assigned.update(loop_nodes)
                emit_escape_edges(
                    children,
                    set(loop_nodes),
                    reason="loop_escape_edge",
                    header=bb,
                    header_false_is_fallthrough=True,
                    loop_header=bb,
                )
                continue

            succ = cfg.succs.get(bb, [])
            merge = cfg.ipdom[bb] if 0 <= bb < len(cfg.ipdom) else -1

            # Minimal switch candidate.  It is conservative: only a multiway CFG
            # branch is serialized as switch, and the body is still recursively
            # validated/fallbacked.
            if not control_depth_exhausted and is_switch_header(bb, region_nodes):
                roots = [s for s in succ if s in region_nodes and s != merge]
                switch_body = _collect_until_merge(
                    cfg,
                    roots,
                    merge,
                    {bb},
                    max_nodes=max_region_nodes,
                )
                switch_body = {x for x in switch_body if x in region_nodes}
                switch_region = {bb} | switch_body
                if switch_body and not (switch_region & assigned) and len(switch_region) <= max_region_nodes:
                    cond = b.slot("COND", [bb], ordered(switch_region), role="switch_cond", owner=f"switch_bb_{bb}")
                    node = Node(
                        kind="SWITCH",
                        slot=cond,
                        children=structure_region(switch_body, role="switch_body", depth=depth + 1),
                        region_bbs=ordered(switch_region),
                    )
                    children.append(node)
                    b.structured_regions.append(set(switch_region))
                    assigned.update(switch_region)
                    emit_escape_edges(children, switch_region, merge=merge, reason="switch_escape_edge")
                    continue

            # Binary branch: if / if-else.  Branches are allowed to join outside
            # the current recursive region; this is what lets an outer fallback
            # body become nested structure on the next pass.
            local_policy = str((if_policy_by_header or {}).get(int(bb), if_policy))
            if (
                not control_depth_exhausted
                and not _if_policy_suppresses_header(local_policy)
                and len(succ) >= 2
                and (
                (merge >= 0 and merge != bb)
                or str((if_policy_by_header or {}).get(int(bb), if_policy)) == "guard_exit"
                )
            ):
                branch_succs = list(succ)
                if merge >= 0 and merge != bb:
                    branch_sets, too_large = branch_sets_for_header(bb, region_nodes, merge)
                else:
                    branch_succs = [int(s) for s in succ if int(s) in region_nodes]
                    branch_sets, too_large = _collect_guard_branch_sets_no_merge(
                        cfg,
                        int(bb),
                        branch_succs,
                        max_region_nodes=max_region_nodes,
                    )
                    branch_sets = [
                        {int(x) for x in nodes if int(x) in region_nodes}
                        for nodes in branch_sets
                    ]
                union = set().union(*branch_sets) if branch_sets else set()
                if (
                    not too_large
                    and union
                    and disjoint_branch_sets(branch_sets)
                ):
                    selected_idx: Optional[int] = None
                    if local_policy == "guard_exit":
                        selected_idx = _preferred_guard_branch_index(
                            cfg,
                            branch_succs,
                            branch_sets,
                            merge,
                            exit_like_nodes,
                        )
                    else:
                        selected_idx = _if_policy_selected_index(local_policy, len(branch_sets))

                    fallthrough_roots: Set[int] = set()
                    early_exit_target = _empty_exit_branch_target(
                        cfg,
                        branch_succs,
                        branch_sets,
                        selected_idx,
                        merge,
                        exit_like_nodes,
                    )
                    if selected_idx is None or (not set(branch_sets[selected_idx]) and early_exit_target is None):
                        nonempty = [set(x) for x in branch_sets if x]
                        if len(nonempty) == 1:
                            then_nodes = nonempty[0]
                            else_nodes: Set[int] = set()
                        else:
                            then_nodes = set(branch_sets[0]) if branch_sets else set()
                            else_nodes = set().union(*branch_sets[1:]) if len(branch_sets) > 1 else set()
                        control_region = {bb} | union
                    else:
                        then_nodes = set(branch_sets[selected_idx])
                        else_nodes = set()
                        control_region = {bb} | then_nodes
                        for i, root in enumerate(branch_succs):
                            if i != selected_idx and root != merge:
                                fallthrough_roots.add(int(root))

                    if (
                        control_region & assigned
                        or len(control_region) > max_region_nodes
                        or len(control_region) < max(1, int(min_branch_region_nodes))
                    ):
                        continue

                    cond = b.slot(
                        "COND",
                        [bb],
                        ordered(control_region),
                        role=f"if_cond_{local_policy}",
                        owner=f"if_bb_{bb}",
                    )
                    node = Node(
                        kind="IF_ELSE" if else_nodes else "IF",
                        slot=cond,
                        children=structure_region(then_nodes, role="then", depth=depth + 1),
                        else_children=structure_region(else_nodes, role="else", depth=depth + 1),
                        region_bbs=ordered(control_region),
                    )
                    if selected_idx is not None and not then_nodes and early_exit_target is not None:
                        node.children = [
                            b.early_exit_slot(
                                (int(bb), int(early_exit_target)),
                                reason="guard_empty_branch_early_exit",
                            )
                        ]
                    children.append(node)
                    b.structured_regions.append(set(control_region))
                    assigned.update(control_region)
                    emit_escape_edges(
                        children,
                        control_region,
                        merge=merge,
                        reason="if_escape_edge",
                        header=bb,
                        header_false_is_fallthrough=bool(fallthrough_roots),
                    )
                    continue

            # Plain node.  Keep it as a fillable block; do not call it
            # unstructured unless a whole region exceeds limits above.
            children.append(b.block([bb], role=role))
            assigned.add(bb)

        return children

    children = structure_region(set(range(cfg.n_bb)), role="bb", depth=0)
    root = Node(kind="FUNCTION", children=children, region_bbs=list(range(cfg.n_bb)))
    cand = Candidate(
        name=name,
        root=root,
        slots=b.slots,
        explicit_edges=b.explicit_edges,
        structured_regions=b.structured_regions,
        fallback_regions=b.fallback_regions,
    )
    _add_missing_edge_gotos(cfg, cand)
    return cand


def _add_missing_edge_gotos(cfg: CFGView, cand: Candidate) -> None:
    edge_report = explain_candidate_edges(cfg, cand)
    uncovered_edges = edge_report.get("uncovered_edges") or []
    if uncovered_edges:
        # Append explicit escape slots at function tail for any edge not
        # explainable by structured semantics, linear slots, or fallback
        # regions.  This makes the MVP conservative: a pretty structured shape
        # may be incomplete, but the CFG edge is still not silently lost.
        # Prefer precise BREAK/CONTINUE/EARLY_EXIT slots when the CFG context is
        # obvious; use opaque GOTO only as the final residual edge kind.
        if cand.root.children is None:
            cand.root.children = []
        next_slot = len(cand.slots)
        early_exit_targets = _function_exit_like_nodes(cfg)

        typed_edges: List[Tuple[Edge, str, str, str, str]] = []
        cleanup_edges_by_target: Dict[int, List[Edge]] = {}
        for raw_edge in uncovered_edges:
            edge = (int(raw_edge[0]), int(raw_edge[1]))
            slot_type = "GOTO"
            role = "goto"
            node_kind = "GOTO_SLOT"
            reason = "validator_added_missing_edge"
            for header, loop_nodes in cfg.loop_nodes_by_header.items():
                loop_set = {int(x) for x in loop_nodes}
                if edge[1] == int(header) and edge[0] in loop_set:
                    slot_type = "CONTINUE"
                    role = "continue"
                    node_kind = "CONTINUE_SLOT"
                    reason = "validator_added_loop_continue_edge"
                    break
                if edge[0] in loop_set and edge[1] not in loop_set:
                    slot_type = "BREAK"
                    role = "break"
                    node_kind = "BREAK_SLOT"
                    reason = "validator_added_loop_break_edge"
                    break
            if slot_type == "GOTO" and edge[1] in early_exit_targets:
                slot_type = "EARLY_EXIT"
                role = "early_exit"
                node_kind = "EARLY_EXIT_SLOT"
                reason = "validator_added_early_exit_edge"
            typed_edges.append((edge, slot_type, role, node_kind, reason))
            if slot_type == "EARLY_EXIT":
                cleanup_edges_by_target.setdefault(edge[1], []).append(edge)

        consumed_cleanup_edges: Set[Edge] = set()
        for target, edges in sorted(cleanup_edges_by_target.items()):
            srcs = _uniq_ints(s for s, _d in edges)
            # Multiple residual edges to the same exit-like BB are a canonical
            # cleanup/return-exit region, not independent opaque gotos.  Keep
            # every CFG edge explicit while exposing a single fillable cleanup
            # slot.
            if len(edges) < 2:
                continue
            slot = Slot(
                slot_id=next_slot,
                slot_type="CLEANUP_EXIT",
                primary_bbs=[],
                context_bbs=srcs + [int(target)],
                role="cleanup_exit",
                owner=f"cleanup_exit_bb_{int(target)}",
                reason="validator_grouped_cleanup_exit_edges",
            )
            next_slot += 1
            cand.slots.append(slot)
            for edge in edges:
                edge = (int(edge[0]), int(edge[1]))
                cand.explicit_edges.add(edge)
                consumed_cleanup_edges.add(edge)
            cand.root.children.append(
                Node(
                    kind="CLEANUP_EXIT_SLOT",
                    slot=slot,
                    region_bbs=srcs + [int(target)],
                    reason=slot.reason,
                )
            )

        for edge, slot_type, role, node_kind, reason in typed_edges:
            edge = (int(edge[0]), int(edge[1]))
            if edge in consumed_cleanup_edges:
                continue
            slot = Slot(
                slot_id=next_slot,
                slot_type=slot_type,
                primary_bbs=[],
                context_bbs=[int(edge[0]), int(edge[1])],
                role=role,
                edge=edge,
                reason=reason,
            )
            next_slot += 1
            cand.slots.append(slot)
            cand.explicit_edges.add(edge)
            cand.root.children.append(
                Node(kind=node_kind, slot=slot, region_bbs=[edge[0], edge[1]], reason=slot.reason)
            )
    _repair_missing_bb_coverage(cfg, cand)


def _repair_missing_bb_coverage(cfg: CFGView, cand: Candidate, *, max_missing: int = 5) -> None:
    report = validate_candidate(cfg, cand, rendered_text=render_candidate(cand))
    if not report.missing_bbs:
        return
    if len(report.missing_bbs) > int(max_missing):
        return
    if report.duplicate_primary_bbs or report.uncovered_edges or report.invalid_bb_refs:
        return
    if [str(x) for x in report.fatal_errors] != ["missing_bb_coverage"]:
        return
    repaired = [int(bb) for bb in sorted(report.missing_bbs)]

    local = deepcopy(cand)
    _apply_missing_bb_repair(cfg, local, repaired, prefer_local=True)
    local_report = validate_candidate(cfg, local, rendered_text=render_candidate(local))
    if _repair_report_ok(local_report):
        _replace_candidate_contents(cand, local)
        return

    tail = deepcopy(cand)
    _apply_missing_bb_repair(cfg, tail, repaired, prefer_local=False)
    tail_report = validate_candidate(cfg, tail, rendered_text=render_candidate(tail))
    if _repair_report_ok(tail_report):
        _replace_candidate_contents(cand, tail)


def _repair_report_ok(report: ValidationReport) -> bool:
    return bool(
        report.valid
        and report.strict_edge_preservation_ok
        and report.bb_coverage_ok
        and report.edge_preservation_ok
        and report.slot_grounding_ok
        and report.parse_success
        and not report.fatal_errors
    )


def _replace_candidate_contents(dst: Candidate, src: Candidate) -> None:
    dst.root = src.root
    dst.slots = src.slots
    dst.explicit_edges = src.explicit_edges
    dst.structured_regions = src.structured_regions
    dst.fallback_regions = src.fallback_regions
    dst.interval_regions = src.interval_regions
    dst.notes = src.notes


def _apply_missing_bb_repair(
    cfg: CFGView,
    cand: Candidate,
    repaired: Sequence[int],
    *,
    prefer_local: bool,
) -> None:
    if cand.root.children is None:
        cand.root.children = []
    next_slot = len(cand.slots)
    for bb in repaired:
        bb = int(bb)
        slot = Slot(
            slot_id=next_slot,
            slot_type="BLOCK",
            primary_bbs=[bb],
            context_bbs=[bb],
            role="missing_bb_repair",
            owner=f"missing_bb_repair_{bb}",
            reason="validator_repaired_missing_bb_coverage",
        )
        next_slot += 1
        cand.slots.append(slot)
        node = Node(
            kind="BLOCK_SLOT",
            slot=slot,
            region_bbs=[bb],
            reason=slot.reason,
        )
        if not prefer_local or not _insert_repair_block_locally(cfg, cand.root, bb, node):
            cand.root.children.append(node)
    cand.notes.append(f"repaired_missing_bb_coverage={repaired}")


def _insert_repair_block_locally(cfg: CFGView, root: Node, bb: int, repair_node: Node) -> bool:
    """Place a validator repair slot inside the smallest owning control node.

    Missing-BB repair is a safety net: it should restore primary BB coverage
    without changing CFG semantics.  Tail-appending every repair block preserves
    hard validity, but it is unnecessarily hostile to Module C when the missing
    BB already belongs to a LOOP/IF/SWITCH region.  In that common case, keep the
    slot local to the control body and fall back to tail insertion only when no
    conservative owner can be identified.
    """

    target = int(bb)
    controls: List[Tuple[int, int, Node]] = []

    def walk(node: Node, depth: int) -> None:
        if node.kind in {"LOOP", "IF", "IF_ELSE", "SWITCH"}:
            region = {int(x) for x in node.region_bbs}
            header = {
                int(x)
                for x in (node.slot.primary_bbs if node.slot is not None else [])
            }
            nested = _children_region(node.children) | _children_region(node.else_children)
            if target in region and target not in header and target not in nested:
                controls.append((len(region), -depth, node))
        for child in node.children:
            walk(child, depth + 1)
        for child in node.else_children:
            walk(child, depth + 1)

    walk(root, 0)
    if not controls:
        return False

    _size, _neg_depth, owner = min(controls, key=lambda x: (x[0], x[1]))
    branch_children = _repair_branch_children(cfg, owner, target)
    branch_children.append(repair_node)
    return True


def _repair_branch_children(cfg: CFGView, owner: Node, bb: int) -> List[Node]:
    if owner.kind != "IF_ELSE" or owner.slot is None or not owner.slot.primary_bbs:
        return owner.children

    header = int(owner.slot.primary_bbs[0])
    succs = [int(s) for s in cfg.succs.get(header, [])]
    if len(succs) < 2:
        return owner.children
    merge = cfg.ipdom[header] if 0 <= header < len(cfg.ipdom) else -1
    if merge < 0 or merge == header:
        return owner.children

    matches: List[int] = []
    for idx, succ in enumerate(succs):
        if succ == merge:
            branch_nodes: Set[int] = set()
        else:
            branch_nodes = _collect_until_merge(
                cfg,
                [succ],
                merge,
                {header},
                max_nodes=max(1, cfg.n_bb),
            )
        if int(bb) in {int(x) for x in branch_nodes}:
            matches.append(idx)

    if len(matches) == 1 and matches[0] > 0 and owner.else_children is not None:
        return owner.else_children
    return owner.children


def _candidate_key(cand: Candidate) -> Tuple[Any, ...]:
    outline = tuple(candidate_outline(cand))
    slot_sig = tuple((s.slot_type, tuple(s.primary_bbs), tuple(s.context_bbs), s.role) for s in cand.slots)
    structured_sig = tuple(sorted(tuple(sorted(int(x) for x in reg)) for reg in cand.structured_regions))
    fallback_sig = tuple(sorted(tuple(sorted(int(x) for x in reg)) for reg in cand.fallback_regions))
    interval_sig = tuple(sorted(tuple(sorted(int(x) for x in reg)) for reg in cand.interval_regions))
    explicit_sig = tuple(sorted((int(s), int(d)) for s, d in cand.explicit_edges))
    return outline, slot_sig, structured_sig, fallback_sig, interval_sig, explicit_sig


def _if_candidate_headers(cfg: CFGView, *, max_region_nodes: int) -> List[int]:
    headers: List[int] = []
    exit_like = _function_exit_like_nodes(cfg)
    for bb in cfg.dfs_order:
        succ = cfg.succs.get(int(bb), [])
        merge = cfg.ipdom[int(bb)] if 0 <= int(bb) < len(cfg.ipdom) else -1
        if len(succ) < 2:
            continue
        if merge >= 0 and merge != int(bb):
            branch_sets: List[Set[int]] = []
            too_large = False
            for s in succ:
                if s == merge:
                    nodes: Set[int] = set()
                else:
                    nodes = _collect_until_merge(
                        cfg,
                        [s],
                        merge,
                        {int(bb)},
                        max_nodes=max_region_nodes,
                    )
                if len(nodes) > max_region_nodes:
                    too_large = True
                branch_sets.append(nodes)
        else:
            branch_sets, too_large = _collect_guard_branch_sets_no_merge(
                cfg,
                int(bb),
                succ,
                max_region_nodes=max_region_nodes,
            )
            if _preferred_guard_branch_index(cfg, succ, branch_sets, merge, exit_like) is None:
                continue
        union = set().union(*branch_sets) if branch_sets else set()
        if not union or too_large:
            continue
        seen: Set[int] = set()
        disjoint = True
        for nodes in branch_sets:
            if seen & nodes:
                disjoint = False
                break
            seen.update(nodes)
        if disjoint:
            headers.append(int(bb))
    return headers


def _branch_policy_options_for_header(
    cfg: CFGView,
    header: int,
    *,
    max_region_nodes: int,
) -> List[str]:
    bb = int(header)
    succs = [int(s) for s in cfg.succs.get(bb, []) if 0 <= int(s) < cfg.n_bb]
    merge = cfg.ipdom[bb] if 0 <= bb < len(cfg.ipdom) else -1
    out: List[str] = ["then_only", "else_only", "guard_exit", "plain"]
    if len(succs) <= 2 or merge < 0 or merge == bb:
        return out
    branch_sets: List[Set[int]] = []
    for s in succs:
        if s == merge:
            branch_sets.append(set())
        else:
            branch_sets.append(
                _collect_until_merge(
                    cfg,
                    [s],
                    merge,
                    {bb},
                    max_nodes=max_region_nodes,
                )
            )
    exit_like = _function_exit_like_nodes(cfg)
    preferred = _preferred_guard_branch_index(cfg, succs, branch_sets, merge, exit_like)
    if preferred is not None:
        out.insert(0, f"only_{int(preferred)}")
    # A "plain" variant lets mixed candidates turn off one over-eager binary
    # branch while keeping other headers structured.  This is a generator-side
    # recall option for source patterns where the binary CFG has an extra
    # compare/join chain that the source AST does not expose as a separate if.
    out.append("plain")
    for idx, nodes in enumerate(branch_sets):
        if idx >= 4:
            break
        if idx < 2:
            continue
        if nodes or _empty_exit_branch_target(cfg, succs, branch_sets, idx, merge, exit_like) is not None:
            out.append(f"only_{int(idx)}")
    dedup: List[str] = []
    seen: Set[str] = set()
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _mixed_if_policy_maps(
    cfg: CFGView,
    headers: Sequence[int],
    *,
    max_mixed_headers: int,
    max_maps: int,
    max_region_nodes: int,
) -> List[Dict[int, str]]:
    hs = [int(h) for h in headers]
    maps: List[Dict[int, str]] = []
    seen_maps: Set[Tuple[Tuple[int, str], ...]] = set()
    options = {
        int(h): _branch_policy_options_for_header(
            cfg,
            int(h),
            max_region_nodes=max_region_nodes,
        )
        for h in hs
    }

    def policy_prefix(h: int, n: int) -> List[str]:
        raw = list(options.get(int(h), ["then_only", "else_only", "guard_exit", "plain"]))
        out: List[str] = []
        for p in raw:
            if p not in out:
                out.append(str(p))
            if len(out) >= int(n):
                break
        if "plain" in raw and "plain" not in out:
            if len(out) >= int(n) and out:
                out[-1] = "plain"
            else:
                out.append("plain")
        return out

    def add_map(m: Dict[int, str]) -> bool:
        key = tuple(sorted((int(h), str(p)) for h, p in m.items()))
        if key in seen_maps:
            return False
        seen_maps.add(key)
        maps.append({int(h): str(p) for h, p in m.items()})
        return len(maps) >= max_maps

    # Give every header at least a chance before spending the beam on early
    # combinations.  This is important for large functions where source-like
    # guards often appear after the first few branch headers in DFS order.
    for h in hs:
        for policy in policy_prefix(int(h), 5):
            if add_map({int(h): str(policy)}):
                return maps

    for width in range(2, min(int(max_mixed_headers), len(hs)) + 1):
        for combo in combinations(hs, width):
            combo_options = [policy_prefix(int(h), 4) for h in combo]
            for choices in product(*combo_options):
                if add_map({int(h): str(p) for h, p in zip(combo, choices)}):
                    return maps
    return maps


def _switch_chain_headers(
    cfg: CFGView,
    *,
    max_region_nodes: int,
    conservative: bool = True,
) -> List[int]:
    headers: List[int] = []
    for bb in cfg.dfs_order:
        if _switch_chain_region(
            cfg,
            int(bb),
            max_region_nodes=max_region_nodes,
            conservative=bool(conservative),
        ) is not None:
            headers.append(int(bb))
    return headers


def _switch_fanin_headers(cfg: CFGView, *, max_region_nodes: int) -> List[int]:
    headers: List[int] = []
    for bb in cfg.dfs_order:
        if _switch_fanin_region(
            cfg,
            int(bb),
            max_region_nodes=max_region_nodes,
        ) is not None:
            headers.append(int(bb))
    return headers


def _append_unique_candidate(out: List[Candidate], seen: Set[Tuple[Any, ...]], cand: Candidate) -> None:
    key = _candidate_key(cand)
    if key in seen:
        return
    seen.add(key)
    out.append(cand)


def generate_candidates(
    cfg: CFGView,
    *,
    max_region_nodes: int = 64,
    max_mixed_headers: int = 3,
    candidate_beam: int = 80,
    include_switch_chain: bool = False,
    include_sese_interval: bool = True,
) -> List[Candidate]:
    if cfg.n_bb <= 0:
        return [_full_fallback_candidate(cfg)]
    out: List[Candidate] = []
    seen: Set[Tuple[Any, ...]] = set()

    for cand in (
        _shallow_structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="auto",
            name="cfg_shallow_mvp",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="auto",
            name="cfg_recursive_mvp",
        ),
        _shallow_structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="then_only",
            name="cfg_shallow_then_only",
        ),
        _shallow_structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="else_only",
            name="cfg_shallow_else_only",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="then_only",
            name="cfg_recursive_then_only",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="else_only",
            name="cfg_recursive_else_only",
        ),
        _shallow_structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="guard_exit",
            name="cfg_shallow_guard_exits",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            if_policy="guard_exit",
            name="cfg_recursive_guard_exits",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            max_control_depth=1,
            if_policy="auto",
            name="cfg_recursive_depth1",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            max_control_depth=2,
            if_policy="auto",
            name="cfg_recursive_depth2",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            min_branch_region_nodes=3,
            if_policy="auto",
            name="cfg_recursive_skip_tiny_if",
        ),
        _structured_candidate(
            cfg,
            max_region_nodes=max_region_nodes,
            min_branch_region_nodes=5,
            if_policy="auto",
            name="cfg_recursive_skip_small_if",
        ),
    ):
        _append_unique_candidate(out, seen, cand)

    widened_region_nodes = max(int(max_region_nodes), min(160, max(96, int(cfg.n_bb))))
    if int(cfg.n_bb) > int(max_region_nodes) and widened_region_nodes > int(max_region_nodes):
        for cand in (
            _structured_candidate(
                cfg,
                max_region_nodes=widened_region_nodes,
                if_policy="auto",
                name=f"cfg_recursive_wide_{int(widened_region_nodes)}",
            ),
            _structured_candidate(
                cfg,
                max_region_nodes=widened_region_nodes,
                if_policy="guard_exit",
                name=f"cfg_recursive_guard_wide_{int(widened_region_nodes)}",
            ),
        ):
            _append_unique_candidate(out, seen, cand)

    if bool(include_sese_interval):
        for cand in (
            _sese_augmented_recursive_candidate(
                cfg,
                max_region_nodes=max_region_nodes,
                name="cfg_recursive_sese_interval",
            ),
            _sese_interval_sequence_candidate(
                cfg,
                max_region_nodes=max_region_nodes,
                name="cfg_sese_interval_seq",
            ),
        ):
            if cand is not None:
                _append_unique_candidate(out, seen, cand)

    if bool(include_switch_chain):
        for idx, header in enumerate(_switch_fanin_headers(cfg, max_region_nodes=max_region_nodes)):
            if len(out) >= max(1, int(candidate_beam)) - 2:
                break
            cand = _switch_fanin_candidate(
                cfg,
                start=int(header),
                max_region_nodes=max_region_nodes,
                name=f"cfg_switch_fanin_{idx}_{int(header)}",
            )
            if cand is not None:
                _append_unique_candidate(out, seen, cand)
        for idx, header in enumerate(
            _switch_chain_headers(
                cfg,
                max_region_nodes=max_region_nodes,
                conservative=True,
            )
        ):
            if len(out) >= max(1, int(candidate_beam)) - 2:
                break
            cand = _switch_chain_candidate(
                cfg,
                start=int(header),
                max_region_nodes=max_region_nodes,
                conservative=True,
                name=f"cfg_switch_chain_{idx}_{int(header)}",
            )
            if cand is not None:
                _append_unique_candidate(out, seen, cand)

    headers = _if_candidate_headers(cfg, max_region_nodes=max_region_nodes)
    for idx, policy_map in enumerate(
        _mixed_if_policy_maps(
            cfg,
            headers,
            max_mixed_headers=max_mixed_headers,
            max_maps=max(0, int(candidate_beam) - len(out) - 2),
            max_region_nodes=max_region_nodes,
        )
    ):
        if len(out) >= max(1, int(candidate_beam)) - 2:
            break
        label = "_".join(f"{h}:{p[0]}" for h, p in sorted(policy_map.items()))
        _append_unique_candidate(
            out,
            seen,
            _structured_candidate(
                cfg,
                max_region_nodes=max_region_nodes,
                if_policy="auto",
                if_policy_by_header=policy_map,
                name=f"cfg_recursive_mixed_{idx}_{label}",
            ),
        )

    if bool(include_switch_chain):
        relaxed_cap = max(2, min(12, int(candidate_beam) // 12))
        for idx, header in enumerate(
            _switch_chain_headers(
                cfg,
                max_region_nodes=max_region_nodes,
                conservative=False,
            )
        ):
            if idx >= relaxed_cap or len(out) >= max(1, int(candidate_beam)) - 2:
                break
            cand = _switch_chain_candidate(
                cfg,
                start=int(header),
                max_region_nodes=max_region_nodes,
                conservative=False,
                name=f"cfg_switch_chain_relaxed_{idx}_{int(header)}",
            )
            if cand is not None:
                _append_unique_candidate(out, seen, cand)

    _append_unique_candidate(out, seen, _linear_candidate(cfg))
    _append_unique_candidate(out, seen, _full_fallback_candidate(cfg))
    return out


def _slot_owners(slots: Sequence[Slot]) -> Dict[int, List[int]]:
    owners: Dict[int, List[int]] = {}
    for s in slots:
        if s.slot_type in _escape_slot_types():
            continue
        for bb in s.primary_bbs:
            owners.setdefault(int(bb), []).append(int(s.slot_id))
    return owners


def _edge_explained_by_region(edge: Edge, regions: Sequence[Set[int]]) -> bool:
    s, d = int(edge[0]), int(edge[1])
    return any(s in reg and d in reg for reg in regions)


def _edge_explained_by_linear_slots(edge: Edge, owners: Dict[int, List[int]], slots: Sequence[Slot]) -> bool:
    s, d = int(edge[0]), int(edge[1])
    so = owners.get(s, [])
    do = owners.get(d, [])
    if not so or not do:
        return False
    # A simple sequence edge is explained when the destination's primary slot is
    # the next primary slot after the source's primary slot.
    return any(ds == ss + 1 for ss in so for ds in do)


def _node_region(node: Node) -> Set[int]:
    return {int(x) for x in node.region_bbs}


def _children_region(children: Sequence[Node]) -> Set[int]:
    out: Set[int] = set()
    for child in children:
        out.update(_node_region(child))
    return out


def _add_reason(out: Dict[Edge, str], edge: Edge, reason: str) -> None:
    out.setdefault((int(edge[0]), int(edge[1])), reason)


def _collect_structured_semantic_edges(cfg: CFGView, node: Node, out: Dict[Edge, str]) -> None:
    """Collect CFG edges explained by structured control constructs.

    This is the MVP's lightweight skeleton-to-CFG lowering check.  It does not
    claim source-AST uniqueness; it only accepts CFG edges that can be explained
    by the rendered control construct's own semantics.  Residual cross-region
    edges must be represented by explicit GOTO slots or fallback regions.
    """
    if node.slot is not None and node.slot.primary_bbs:
        header = int(node.slot.primary_bbs[0])
        region = _node_region(node)
        then_region = _children_region(node.children)
        else_region = _children_region(node.else_children)
        body_region = then_region | else_region
        merge = cfg.ipdom[header] if 0 <= header < len(cfg.ipdom) else -1

        if node.kind in {"IF", "IF_ELSE"}:
            for edge in cfg.edges:
                s, d = int(edge[0]), int(edge[1])
                if s == header and d in then_region:
                    _add_reason(out, edge, "if_then_edge")
                elif s == header and d in else_region:
                    _add_reason(out, edge, "if_else_edge")
                elif s == header and merge >= 0 and d == merge and d not in region:
                    _add_reason(out, edge, "if_empty_or_false_edge")
                elif s in body_region and merge >= 0 and d == merge and d not in region:
                    _add_reason(out, edge, "if_join_edge")

        elif node.kind == "LOOP":
            for edge in cfg.edges:
                s, d = int(edge[0]), int(edge[1])
                if s == header and d in body_region:
                    _add_reason(out, edge, "loop_body_edge")
                elif s == header and d not in region:
                    _add_reason(out, edge, "loop_exit_edge")
                elif s in body_region and d == header:
                    _add_reason(out, edge, "loop_backedge")

        elif node.kind == "SWITCH":
            for edge in cfg.edges:
                s, d = int(edge[0]), int(edge[1])
                if s == header and d in body_region:
                    _add_reason(out, edge, "switch_case_edge")
                elif s in body_region and merge >= 0 and d == merge and d not in region:
                    _add_reason(out, edge, "switch_join_edge")
                elif s in body_region and d not in region and len(cfg.preds.get(d, [])) >= 3:
                    _add_reason(out, edge, "switch_join_edge")

    for child in node.children:
        _collect_structured_semantic_edges(cfg, child, out)
    for child in node.else_children:
        _collect_structured_semantic_edges(cfg, child, out)


def explain_candidate_edges(cfg: CFGView, cand: Candidate) -> Dict[str, Any]:
    explicit_escape_edges = {
        (int(slot.edge[0]), int(slot.edge[1]))
        for slot in cand.slots
        if slot.slot_type in _escape_slot_types() and slot.edge is not None
    }
    goto_edges = {
        (int(slot.edge[0]), int(slot.edge[1]))
        for slot in cand.slots
        if slot.slot_type == "GOTO" and slot.edge is not None
    }
    break_edges = {
        (int(slot.edge[0]), int(slot.edge[1]))
        for slot in cand.slots
        if slot.slot_type == "BREAK" and slot.edge is not None
    }
    continue_edges = {
        (int(slot.edge[0]), int(slot.edge[1]))
        for slot in cand.slots
        if slot.slot_type == "CONTINUE" and slot.edge is not None
    }
    early_exit_edges = {
        (int(slot.edge[0]), int(slot.edge[1]))
        for slot in cand.slots
        if slot.slot_type == "EARLY_EXIT" and slot.edge is not None
    }
    cleanup_exit_edges = {
        (int(src), int(slot.context_bbs[-1]))
        for slot in cand.slots
        if slot.slot_type == "CLEANUP_EXIT" and len(slot.context_bbs) >= 2
        for src in slot.context_bbs[:-1]
    }
    explicit_edges = {(int(s), int(d)) for s, d in cand.explicit_edges}
    explicit_non_goto = explicit_edges - explicit_escape_edges - cleanup_exit_edges
    fallback_regions = [set(int(x) for x in reg) for reg in cand.fallback_regions]
    owners = _slot_owners(cand.slots)
    structured_semantic: Dict[Edge, str] = {}
    _collect_structured_semantic_edges(cfg, cand.root, structured_semantic)

    categories: Dict[str, List[Edge]] = {
        "goto_slot": [],
        "break_slot": [],
        "continue_slot": [],
        "early_exit_slot": [],
        "cleanup_exit_slot": [],
        "explicit_structured": [],
        "interval_internal": [],
        "fallback_internal": [],
        "linear_slot": [],
        "structured_semantic": [],
        "uncovered": [],
    }
    semantic_reasons: Dict[str, int] = {}
    for raw_edge in cfg.edges:
        edge = (int(raw_edge[0]), int(raw_edge[1]))
        if edge in goto_edges:
            categories["goto_slot"].append(edge)
        elif edge in break_edges:
            categories["break_slot"].append(edge)
        elif edge in continue_edges:
            categories["continue_slot"].append(edge)
        elif edge in early_exit_edges:
            categories["early_exit_slot"].append(edge)
        elif edge in cleanup_exit_edges:
            categories["cleanup_exit_slot"].append(edge)
        elif edge in explicit_non_goto:
            categories["explicit_structured"].append(edge)
        elif _edge_explained_by_region(edge, cand.interval_regions):
            categories["interval_internal"].append(edge)
        elif _edge_explained_by_region(edge, fallback_regions):
            categories["fallback_internal"].append(edge)
        elif _edge_explained_by_linear_slots(edge, owners, cand.slots):
            categories["linear_slot"].append(edge)
        elif edge in structured_semantic:
            categories["structured_semantic"].append(edge)
            reason = structured_semantic[edge]
            semantic_reasons[reason] = semantic_reasons.get(reason, 0) + 1
        else:
            categories["uncovered"].append(edge)

    counts = {k: len(v) for k, v in categories.items()}
    preview = {
        k: [[int(s), int(d)] for s, d in v[:24]]
        for k, v in categories.items()
        if v
    }
    explained = len(cfg.edges) - counts["uncovered"]
    return {
        "counts": counts,
        "preview": preview,
        "semantic_reasons": semantic_reasons,
        "category_edges": {
            k: [[int(s), int(d)] for s, d in v]
            for k, v in categories.items()
        },
        "uncovered_edges": list(categories["uncovered"]),
        "edge_preservation_ok": counts["uncovered"] == 0,
        "edge_preservation_ratio": float(explained / max(1, len(cfg.edges))),
    }


def validate_candidate(cfg: CFGView, cand: Candidate, *, rendered_text: str = "") -> ValidationReport:
    fatal: List[str] = []
    invalid_refs: List[Dict[str, Any]] = []
    grounded = True
    for s in cand.slots:
        for key, vals in (("primary_bbs", s.primary_bbs), ("context_bbs", s.context_bbs)):
            for bb in vals:
                if bb < 0 or bb >= cfg.n_bb:
                    invalid_refs.append({"slot_id": s.slot_id, "key": key, "bb": bb})
        if (
            s.slot_type not in _escape_slot_types()
            and not s.primary_bbs
            and not (cfg.n_bb == 0 and s.slot_type == "UNSTRUCTURED_REGION")
        ):
            grounded = False
        if (
            not s.context_bbs
            and s.slot_type not in _escape_slot_types()
            and not (cfg.n_bb == 0 and s.slot_type == "UNSTRUCTURED_REGION")
        ):
            grounded = False

    owners = _slot_owners(cand.slots)
    missing = [bb for bb in range(cfg.n_bb) if bb not in owners]
    dup = {bb: ids for bb, ids in owners.items() if len(ids) > 1}
    bb_coverage_ok = not missing
    unique_primary_ok = not dup
    slot_grounding_ok = grounded and not invalid_refs

    edge_report = explain_candidate_edges(cfg, cand)
    uncovered: List[Edge] = [
        (int(edge[0]), int(edge[1]))
        for edge in edge_report.get("uncovered_edges", [])
    ]
    edge_ok = not uncovered

    parse_success = True
    if rendered_text:
        parse_success = rendered_text.count("{") == rendered_text.count("}")
        if not parse_success:
            fatal.append("unbalanced_braces")

    if invalid_refs:
        fatal.append("invalid_bb_ref")
    if not bb_coverage_ok:
        fatal.append("missing_bb_coverage")
    if not unique_primary_ok:
        fatal.append("duplicate_primary_bb")
    if not edge_ok:
        fatal.append("uncovered_cfg_edge")
    if not slot_grounding_ok:
        fatal.append("slot_grounding_incomplete")

    fallback_bbs = set().union(*cand.fallback_regions) if cand.fallback_regions else set()
    structured_bbs = set().union(*cand.structured_regions) if cand.structured_regions else set()
    goto_count = sum(1 for s in cand.slots if s.slot_type == "GOTO")
    break_count = sum(1 for s in cand.slots if s.slot_type == "BREAK")
    continue_count = sum(1 for s in cand.slots if s.slot_type == "CONTINUE")
    early_exit_count = sum(1 for s in cand.slots if s.slot_type == "EARLY_EXIT")
    cleanup_exit_count = sum(1 for s in cand.slots if s.slot_type == "CLEANUP_EXIT")
    escape_count = goto_count + break_count + continue_count + early_exit_count + cleanup_exit_count
    structured_control_count = _count_structured_controls(cand.root)
    locality = _slot_locality_report(cand)
    score = (
        1000.0 * float(bb_coverage_ok and edge_ok and slot_grounding_ok and parse_success)
        + 20.0 * structured_control_count
        + 0.5 * len(structured_bbs)
        + 8.0 * float(locality["slot_locality_score"])
        - 2.0 * len(fallback_bbs)
        - 0.2 * goto_count
        - 0.08 * break_count
        - 0.05 * continue_count
        - 0.04 * early_exit_count
        - 0.03 * cleanup_exit_count
        - 12.0 * int(locality["empty_control_count"])
        - 30.0 * int(locality["empty_control_with_context_count"])
        - 3.0 * int(locality["body_bb_outside_control_count"])
        - 100.0 * len(fatal)
    )
    valid = bb_coverage_ok and unique_primary_ok and edge_ok and slot_grounding_ok and parse_success and not invalid_refs

    return ValidationReport(
        valid=valid,
        bb_coverage_ok=bb_coverage_ok,
        unique_primary_ok=unique_primary_ok,
        edge_preservation_ok=edge_ok,
        slot_grounding_ok=slot_grounding_ok,
        parse_success=parse_success,
        fatal_errors=fatal,
        missing_bbs=missing,
        duplicate_primary_bbs=dup,
        uncovered_edges=uncovered,
        invalid_bb_refs=invalid_refs,
        edge_explanation_counts=dict(edge_report.get("counts") or {}),
        edge_explanation_preview=dict(edge_report.get("preview") or {}),
        strict_edge_preservation_ok=bool(edge_report.get("edge_preservation_ok")),
        fallback_region_count=len(cand.fallback_regions),
        fallback_bb_count=len(fallback_bbs),
        goto_count=goto_count,
        break_count=break_count,
        continue_count=continue_count,
        early_exit_count=early_exit_count,
        cleanup_exit_count=cleanup_exit_count,
        escape_count=escape_count,
        structured_control_count=structured_control_count,
        structured_bb_count=len(structured_bbs),
        slot_count=len(cand.slots),
        slot_locality_score=float(locality["slot_locality_score"]),
        empty_control_count=int(locality["empty_control_count"]),
        empty_loop_count=int(locality["empty_loop_count"]),
        empty_if_count=int(locality["empty_if_count"]),
        empty_switch_count=int(locality["empty_switch_count"]),
        empty_control_with_context_count=int(locality["empty_control_with_context_count"]),
        control_body_total_bb_count=int(locality["control_body_total_bb_count"]),
        control_body_nested_bb_count=int(locality["control_body_nested_bb_count"]),
        body_bb_outside_control_count=int(locality["body_bb_outside_control_count"]),
        score=score,
    )


def _count_structured_controls(node: Node) -> int:
    n = 1 if node.kind in {"IF", "IF_ELSE", "LOOP", "SWITCH"} else 0
    for c in node.children:
        n += _count_structured_controls(c)
    for c in node.else_children:
        n += _count_structured_controls(c)
    return n


def _slot_locality_report(cand: Candidate) -> Dict[str, Any]:
    """Measure whether fillable BB slots are nested under owning controls.

    This is intentionally a quality signal rather than a CFG-validity rule.
    A candidate can preserve every binary edge while still being bad for Module C
    if a LOOP/IF/SWITCH claims a region but renders that region's block slots
    outside the control body.
    """

    empty_by_kind: Dict[str, int] = {"LOOP": 0, "IF": 0, "IF_ELSE": 0, "SWITCH": 0}
    empty_with_context = 0
    body_total = 0
    body_nested = 0
    outside = 0

    def primary_region(node: Node) -> Set[int]:
        if node.slot is not None and node.slot.slot_type not in _escape_slot_types():
            return {int(x) for x in node.slot.primary_bbs}
        out: Set[int] = set()
        for child in node.children:
            out.update(_node_region(child))
        for child in node.else_children:
            out.update(_node_region(child))
        return out

    def walk(node: Node) -> None:
        nonlocal empty_with_context, body_total, body_nested, outside
        if node.kind in {"LOOP", "IF", "IF_ELSE", "SWITCH"}:
            child_region = _children_region(node.children) | _children_region(node.else_children)
            header = {int(x) for x in (node.slot.primary_bbs if node.slot is not None else [])}
            region = _node_region(node)
            expected_body = set(region) - set(header)
            if not child_region:
                empty_by_kind[node.kind] = empty_by_kind.get(node.kind, 0) + 1
                if expected_body:
                    empty_with_context += 1
            body_total += len(expected_body)
            nested_here = len(expected_body & child_region)
            body_nested += nested_here
            outside += max(0, len(expected_body - child_region))

        for child in node.children:
            walk(child)
        for child in node.else_children:
            walk(child)

    walk(cand.root)

    total_empty = sum(empty_by_kind.values())
    score = float(body_nested / max(1, body_total))
    if body_total == 0 and total_empty == 0:
        score = 1.0
    return {
        "slot_locality_score": score,
        "empty_control_count": int(total_empty),
        "empty_loop_count": int(empty_by_kind.get("LOOP", 0)),
        "empty_if_count": int(empty_by_kind.get("IF", 0) + empty_by_kind.get("IF_ELSE", 0)),
        "empty_switch_count": int(empty_by_kind.get("SWITCH", 0)),
        "empty_control_with_context_count": int(empty_with_context),
        "control_body_total_bb_count": int(body_total),
        "control_body_nested_bb_count": int(body_nested),
        "body_bb_outside_control_count": int(outside),
    }


def _collect_outline(node: Node, depth: int, out: List[Tuple[str, int]]) -> None:
    if node.kind in {"IF", "IF_ELSE"}:
        out.append(("if", depth))
        for c in node.children:
            _collect_outline(c, depth + 1, out)
        for c in node.else_children:
            _collect_outline(c, depth + 1, out)
        return
    if node.kind == "LOOP":
        out.append(("loop", depth))
        for c in node.children:
            _collect_outline(c, depth + 1, out)
        return
    if node.kind == "SWITCH":
        out.append(("switch", depth))
        for c in node.children:
            _collect_outline(c, depth + 1, out)
        return
    for c in node.children:
        _collect_outline(c, depth, out)


def candidate_outline(cand: Candidate) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    _collect_outline(cand.root, 0, out)
    return out


def _render_slot_ref(slot: Slot) -> str:
    return f"slot_{int(slot.slot_id):03d}"


def _render_slot_comment(slot: Slot) -> str:
    parts = [
        f"primary {_fmt_bbs(slot.primary_bbs)}",
        f"context {_fmt_bbs(slot.context_bbs)}",
    ]
    if slot.edge is not None:
        parts.append(f"edge {_fmt_edge(slot.edge)}")
    if slot.reason:
        parts.append(f"reason={slot.reason}")
    return "  /* " + "; ".join(parts) + " */"


def render_candidate(cand: Candidate, *, decl_line: str = "int func(...)") -> str:
    lines: List[str] = [f"{decl_line.rstrip().rstrip('{').strip()} {{"]

    def emit(node: Node, indent: int) -> None:
        pad = "    " * indent
        if node.kind == "FUNCTION":
            for child in node.children:
                emit(child, indent)
            return
        if node.kind == "SEQ":
            for child in node.children:
                emit(child, indent)
            return
        if node.kind == "BLOCK_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<FILL_BLOCK:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "CASE_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<CASE_SLOT:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "UNSTRUCTURED_REGION" and node.slot is not None:
            lines.append(
                f"{pad}<UNSTRUCTURED_REGION:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "GOTO_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<GOTO_SLOT:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "BREAK_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<BREAK_SLOT:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "CONTINUE_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<CONTINUE_SLOT:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "EARLY_EXIT_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<EARLY_EXIT_SLOT:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "CLEANUP_EXIT_SLOT" and node.slot is not None:
            lines.append(
                f"{pad}<CLEANUP_EXIT_SLOT:{_render_slot_ref(node.slot)}>;"
                f"{_render_slot_comment(node.slot)}"
            )
            return
        if node.kind == "LOOP" and node.slot is not None:
            lines.append(
                f"{pad}while (<FILL_COND:{_render_slot_ref(node.slot)}>) {{"
                f"{_render_slot_comment(node.slot)}"
            )
            for child in node.children:
                emit(child, indent + 1)
            lines.append(f"{pad}}}")
            return
        if node.kind == "SWITCH" and node.slot is not None:
            lines.append(
                f"{pad}switch (<FILL_COND:{_render_slot_ref(node.slot)}>) {{"
                f"{_render_slot_comment(node.slot)}"
            )
            for child in node.children:
                emit(child, indent + 1)
            lines.append(f"{pad}}}")
            return
        if node.kind in {"IF", "IF_ELSE"} and node.slot is not None:
            lines.append(
                f"{pad}if (<FILL_COND:{_render_slot_ref(node.slot)}>) {{"
                f"{_render_slot_comment(node.slot)}"
            )
            for child in node.children:
                emit(child, indent + 1)
            if node.else_children:
                lines.append(f"{pad}}} else {{")
                for child in node.else_children:
                    emit(child, indent + 1)
            lines.append(f"{pad}}}")
            return
        lines.append(f"{pad}/* unsupported node kind={node.kind}; rendered as fallback comment */")

    emit(cand.root, 1)
    lines.append("}")
    return "\n".join(lines)


def select_candidate(cfg: CFGView, candidates: Sequence[Candidate]) -> Tuple[Candidate, ValidationReport, str]:
    best: Optional[Tuple[float, int, Candidate, ValidationReport, str]] = None
    for idx, cand in enumerate(candidates):
        text = render_candidate(cand)
        report = validate_candidate(cfg, cand, rendered_text=text)
        # Prefer valid candidates.  Among valid candidates, prefer structure and
        # less fallback/goto.  Full fallback remains the guaranteed last resort.
        rank = report.score
        if report.valid:
            rank += 10000.0
        if best is None or rank > best[0]:
            best = (rank, -idx, cand, report, text)
    if best is None:
        cand = _full_fallback_candidate(cfg)
        text = render_candidate(cand)
        return cand, validate_candidate(cfg, cand, rendered_text=text), text
    _rank, _idx, cand, _report, text = best
    # Revalidate with the final rendered text.
    return cand, validate_candidate(cfg, cand, rendered_text=text), text


def build_verified_skeleton(
    row: Dict[str, Any],
    *,
    decl_line: str = "int func(...)",
    max_region_nodes: int = 64,
    max_mixed_headers: int = 3,
    candidate_beam: int = 80,
) -> Dict[str, Any]:
    cfg = analyse_cfg(row)
    candidates = generate_candidates(
        cfg,
        max_region_nodes=max_region_nodes,
        max_mixed_headers=max_mixed_headers,
        candidate_beam=candidate_beam,
    )
    cand, report, _ = select_candidate(cfg, candidates)
    text = render_candidate(cand, decl_line=decl_line)
    report = validate_candidate(cfg, cand, rendered_text=text)
    return {
        "candidate_name": cand.name,
        "skeleton": text,
        "slots": [s.asdict() for s in cand.slots],
        "outline": [[k, d] for k, d in candidate_outline(cand)],
        "validator_report": report.asdict(),
        "cfg": {
            "n_bb": cfg.n_bb,
            "edges": [[s, d] for s, d in cfg.edges],
            "dfs_order": list(cfg.dfs_order),
            "unreachable": list(cfg.unreachable),
            "back_edges": [[s, d] for s, d in cfg.back_edges],
            "loop_headers": sorted(int(x) for x in cfg.loop_nodes_by_header.keys()),
        },
        "candidates": [
            {
                "name": c.name,
                "outline": [[k, d] for k, d in candidate_outline(c)],
                "validator_report": validate_candidate(cfg, c, rendered_text=render_candidate(c)).asdict(),
            }
            for c in candidates
        ],
    }
