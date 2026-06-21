"""
Control Structure Tree (CST) construction and skeleton serialisation.

Builds a hierarchical representation of the binary CFG's control flow and
converts it to a code skeleton with ``<FILL_*>`` placeholders that Module C
(the LLM decoder) fills in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from struclift.data.structures import PatternType, Skeleton, SlotInfo
from struclift.utils.graph_utils import compute_dominator_tree


# ═══════════════════════════ CSTNode dataclass ═══════════════════════════════

@dataclass
class CSTNode:
    """A node in the Control Structure Tree.

    Attributes:
        type:         One of ``'function'``, ``'while_loop'``, ``'for_loop'``,
                      ``'if_else'``, ``'switch'``, ``'statement'``.
        children:     Ordered child nodes.
        condition_bb: Index of the basic block that evaluates the condition
                      (for loops, if-else, switch).  ``None`` for statements
                      and function root.
        aligned_bbs:  List of basic-block indices that this CST node
                      corresponds to (may span multiple BBs for compound
                      regions).
        pattern_label: Per-node PatternType int for slot type differentiation.
        branch_label: Optional parent-branch hint.  This keeps nested control
                      structures in an ``else`` branch from being serialized as
                      part of the parent's ``then`` branch.
    """
    type: str
    children: List[CSTNode] = field(default_factory=list)
    condition_bb: Optional[int] = None
    aligned_bbs: List[int] = field(default_factory=list)
    pattern_label: Optional[int] = None
    branch_label: Optional[int] = None


@dataclass
class _RegionInfo:
    """Structured CFG region headed by a control basic block."""

    header: int
    type: str
    pattern_label: int
    condition_bbs: List[int]
    nodes: Set[int]
    merge: int = -1
    then_nodes: Set[int] = field(default_factory=set)
    else_nodes: Set[int] = field(default_factory=set)


def _normalise_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src = [int(x) for x in edge_index[0].detach().cpu().tolist()]
    dst = [int(x) for x in edge_index[1].detach().cpu().tolist()]
    pairs = [
        (s, d) for s, d in zip(src, dst)
        if 0 <= s < num_nodes and 0 <= d < num_nodes
    ]
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([[s for s, _ in pairs], [d for _, d in pairs]], dtype=torch.long)


def _succ_pred_lists(edge_index: torch.Tensor, num_nodes: int) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    succs: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    preds: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            succs[int(s)].append(int(d))
            preds[int(d)].append(int(s))
    for i in range(num_nodes):
        succs[i] = sorted(set(succs[i]))
        preds[i] = sorted(set(preds[i]))
    return succs, preds


def _dfs_order_from_entry(succs: Dict[int, List[int]], num_nodes: int) -> Tuple[List[int], int]:
    order: List[int] = []
    visited: Set[int] = set()
    stack: List[int] = [0] if num_nodes > 0 else []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for succ in reversed(succs.get(node, [])):
            if succ not in visited:
                stack.append(succ)
    n_unreachable = 0
    for i in range(num_nodes):
        if i not in visited:
            order.append(i)
            n_unreachable += 1
    return order, n_unreachable


def _order_positions_from_keys(
    dfs_order: Sequence[int],
    num_nodes: int,
    bb_order_key: Optional[Sequence[object] | Dict[int, object]] = None,
) -> Dict[int, Tuple[float, float, int]]:
    """Return a stable BB ordering map.

    The no-source inference path passes ``None`` and gets the original DFS
    order.  Training/repack paths may pass source-alignment-derived keys so
    sibling CST events follow source statement order when available.
    """
    dfs_pos: Dict[int, int] = {int(bb): i for i, bb in enumerate(dfs_order)}
    out: Dict[int, Tuple[float, float, int]] = {}
    for bb in range(num_nodes):
        fallback = float(dfs_pos.get(bb, bb))
        key_obj = None
        if isinstance(bb_order_key, dict):
            key_obj = bb_order_key.get(bb)
        elif bb_order_key is not None and bb < len(bb_order_key):
            key_obj = bb_order_key[bb]

        parsed: Optional[Tuple[float, float]] = None
        if isinstance(key_obj, (list, tuple)):
            nums: List[float] = []
            for item in key_obj:
                try:
                    nums.append(float(item))
                except Exception:
                    break
                if len(nums) >= 2:
                    break
            if nums:
                parsed = (nums[0], nums[1] if len(nums) > 1 else 0.0)
        elif key_obj is not None:
            try:
                parsed = (float(key_obj), 0.0)
            except Exception:
                parsed = None

        if parsed is None:
            out[bb] = (fallback, 0.0, int(fallback))
        else:
            out[bb] = (parsed[0], parsed[1], int(fallback))
    return out


def _immediate_post_dominators(
    edge_index: torch.Tensor,
    succs: Dict[int, List[int]],
    num_nodes: int,
) -> List[int]:
    """Return each node's immediate post-dominator, or ``-1`` if none is real."""
    if num_nodes <= 0:
        return []
    exits = [i for i in range(num_nodes) if not succs.get(i)]
    if not exits:
        return [-1] * num_nodes

    virtual_exit = num_nodes
    rev_src: List[int] = []
    rev_dst: List[int] = []
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            rev_src.append(int(d))
            rev_dst.append(int(s))
    for e in exits:
        rev_src.append(virtual_exit)
        rev_dst.append(int(e))
    rev = torch.tensor([rev_src, rev_dst], dtype=torch.long) if rev_src else torch.zeros(2, 0, dtype=torch.long)
    idom = compute_dominator_tree(rev, num_nodes + 1, entry_node=virtual_exit).tolist()
    out = [-1] * num_nodes
    for i in range(num_nodes):
        dom = int(idom[i]) if i < len(idom) else -1
        out[i] = dom if 0 <= dom < num_nodes else -1
    return out


# ═══════════════════════ CST construction ════════════════════════════════════

def build_control_structure_tree_with_stats(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    bb_order_key: Optional[Sequence[object] | Dict[int, object]] = None,
    alignment_matrix: Optional[torch.Tensor] = None,
) -> Tuple[CSTNode, Dict[str, int]]:
    """同 :func:`build_control_structure_tree`，但额外返回建树过程统计 ``stats``。

    ``stats`` 字段:
        - ``n_nodes``: 总 BB 数。
        - ``n_headers``: 命中 LOOP/IF/SWITCH_HEADER 的 BB 数。
        - ``n_body_unattached``: body/exit 标签找不到兼容 header、回退挂到当前
          scope 的次数（建树健康度的主要惩罚项）。
        - ``n_loop_exit_orphan``: 出现 LOOP_EXIT 但当前栈没有 open loop 的次数。
        - ``n_unreachable``: 从节点 0 出发 DFS 未访问到的孤立 BB 数。

    与 :func:`compute_pattern_reliability` 一起用于「骨架可靠性分数 R」。
    """
    root, stats = _build_cst_impl(
        pattern_labels,
        edge_index,
        num_nodes,
        bb_order_key,
        alignment_matrix=alignment_matrix,
    )
    return root, stats


def build_control_structure_tree(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    bb_order_key: Optional[Sequence[object] | Dict[int, object]] = None,
    alignment_matrix: Optional[torch.Tensor] = None,
) -> CSTNode:
    """Build a Control Structure Tree from pattern labels and CFG topology.

    The algorithm walks the CFG in DFS order, identifies structural regions
    via the per-node ``pattern_labels``, and nests them into a tree:

    * ``LOOP_HEADER`` -> ``while_loop`` (or ``for_loop`` if a recognisable
      increment exists -- defaults to ``while_loop``).
    * ``IF_HEADER`` -> ``if_else``.
    * ``SWITCH_HEADER`` -> ``switch``.
    * ``SEQUENTIAL``, body nodes, and everything else -> ``statement``.

    The root is always a ``function`` node.

    Args:
        pattern_labels: Per-node pattern label.  Shape ``[num_nodes]`` (int64).
        edge_index:     Edge endpoints.  Shape ``[2, num_edges]`` (int64).
        num_nodes:      Number of nodes.

    Returns:
        Root ``CSTNode`` of type ``'function'``.
    """
    root, _ = _build_cst_impl(
        pattern_labels,
        edge_index,
        num_nodes,
        bb_order_key,
        alignment_matrix=alignment_matrix,
    )
    return root


def _build_cst_impl(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    bb_order_key: Optional[Sequence[object] | Dict[int, object]] = None,
    alignment_matrix: Optional[torch.Tensor] = None,
) -> Tuple[CSTNode, Dict[str, int]]:
    """与 :func:`build_control_structure_tree` 同语义；额外累加建树健康度统计。

    Older versions used a pure DFS stack.  That is too weak for binary CFGs:
    after an ``if`` body reaches its join block, sequential code was still
    attached under the last open header, producing artifacts like
    ``if (c) { if (c) { ... } }``.  The builder below first recovers structured
    regions from CFG post-dominators/natural loops, then serializes nested
    regions in execution order.  Module A still provides only per-BB pattern
    labels; this step is the deterministic "label + CFG -> skeleton" part.
    """
    labels: List[int] = [int(x) for x in pattern_labels.tolist()]
    edge_index = _normalise_edge_index(edge_index.detach().cpu().to(torch.long), num_nodes)
    succs, preds = _succ_pred_lists(edge_index, num_nodes)
    dfs_order, n_unreachable = _dfs_order_from_entry(succs, num_nodes)
    order_pos = _order_positions_from_keys(dfs_order, num_nodes, bb_order_key)

    align_col: List[int] = [-1] * num_nodes
    align_mass: List[float] = [0.0] * num_nodes
    if alignment_matrix is not None and alignment_matrix.numel() > 0:
        try:
            M = alignment_matrix.detach().to(torch.float32).cpu()
            if M.dim() == 2 and M.size(0) >= num_nodes and M.size(1) > 0:
                rows = M[:num_nodes]
                argmax = rows.argmax(dim=1).tolist()
                mass = rows.sum(dim=1).tolist()
                align_col = [int(x) for x in argmax]
                align_mass = [float(x) for x in mass]
                for i, m in enumerate(align_mass):
                    if m <= 1e-12:
                        align_col[i] = -1
        except Exception:
            align_col = [-1] * num_nodes
            align_mass = [0.0] * num_nodes

    def source_col(bb: int) -> int:
        if 0 <= int(bb) < len(align_col):
            return int(align_col[int(bb)])
        return -1

    def source_col_set(nodes: Set[int]) -> Set[int]:
        return {
            source_col(bb)
            for bb in nodes
            if source_col(bb) >= 0
        }

    def source_col_range(nodes: Set[int]) -> Optional[Tuple[int, int]]:
        cols = source_col_set(nodes)
        if not cols:
            return None
        return min(cols), max(cols)

    def source_col_center(nodes: Set[int]) -> Optional[float]:
        cols = [source_col(bb) for bb in nodes if source_col(bb) >= 0]
        if not cols:
            return None
        return float(sum(cols)) / float(len(cols))

    def source_col_distance(bb: int, anchor: int) -> Optional[int]:
        cb, ca = source_col(bb), source_col(anchor)
        if cb < 0 or ca < 0:
            return None
        return abs(cb - ca)

    def source_range_contains(range_: Optional[Tuple[int, int]], bb: int) -> bool:
        col = source_col(bb)
        return range_ is not None and col >= range_[0] and col <= range_[1]

    n_demoted_if_headers = 0
    n_demoted_switch_headers = 0
    n_demoted_loop_headers = 0
    n_promoted_switch_headers = 0
    for bb, lab in enumerate(list(labels)):
        succ = [s for s in succs.get(bb, []) if s != bb]
        pred = [p for p in preds.get(bb, []) if p != bb]
        if lab == int(PatternType.IF_HEADER):
            # A source-level if needs a real CFG branch.  Single-successor
            # IF_HEADER blocks are usually routing/join noise.  Conversely, a
            # multiway branch is stronger binary evidence for a switch than for
            # a chain of independent ifs.
            if len(succ) >= 3:
                labels[bb] = int(PatternType.SWITCH_HEADER)
                n_promoted_switch_headers += 1
            elif len(succ) < 2:
                labels[bb] = int(PatternType.SEQUENTIAL)
                n_demoted_if_headers += 1
        elif lab == int(PatternType.SWITCH_HEADER):
            if len(succ) < 2:
                labels[bb] = int(PatternType.SEQUENTIAL)
                n_demoted_switch_headers += 1
        elif lab == int(PatternType.LOOP_HEADER):
            has_backedge_shape = any(p >= bb for p in pred) or any(s <= bb for s in succ)
            if not has_backedge_shape and len(succ) < 2:
                labels[bb] = int(PatternType.SEQUENTIAL)
                n_demoted_loop_headers += 1

    stats: Dict[str, int] = {
        "n_nodes": int(num_nodes),
        "n_headers": sum(
            1 for x in labels
            if x in {
                int(PatternType.LOOP_HEADER),
                int(PatternType.IF_HEADER),
                int(PatternType.SWITCH_HEADER),
            }
        ),
        "n_body_unattached": 0,
        "n_loop_exit_orphan": 0,
        "n_unreachable": n_unreachable,
        "n_demoted_if_headers": n_demoted_if_headers,
        "n_demoted_switch_headers": n_demoted_switch_headers,
        "n_demoted_loop_headers": n_demoted_loop_headers,
        "n_promoted_switch_headers": n_promoted_switch_headers,
        "n_collapsed_if_header_chains": 0,
        "n_alignment_collapsed_if_headers": 0,
        "n_alignment_refined_if_regions": 0,
        "n_alignment_refined_if_branches": 0,
        "n_alignment_parent_tiebreaks": 0,
        "n_alignment_header_insufficient": 0,
        "n_lifted_nonexclusive_if_children": 0,
        "n_pruned_if_continuation_branches": 0,
        "n_flattened_if_else_chains": 0,
        "n_protected_structured_branches": 0,
    }

    root = CSTNode(type="function", aligned_bbs=list(range(num_nodes)))
    if num_nodes <= 0:
        return root, stats

    idom = compute_dominator_tree(edge_index, num_nodes, entry_node=0).tolist()
    ipdom = _immediate_post_dominators(edge_index, succs, num_nodes)

    def order_key(bb: int) -> Tuple[float, float, int]:
        return order_pos.get(int(bb), (float(bb), 0.0, int(bb)))

    def region_order_key(reg: _RegionInfo) -> Tuple[float, float, int]:
        # Prefer the earliest aligned source column among the condition BBs.
        # This is the least invasive way to use M for CST serialization:
        # siblings become source-like without changing which controls exist.
        cols = [
            source_col(bb)
            for bb in (reg.condition_bbs or [reg.header])
            if source_col(bb) >= 0
        ]
        if cols:
            base = min(cols)
            return (float(base), float(source_col(reg.header) if source_col(reg.header) >= 0 else base), int(order_key(reg.header)[2]))
        return order_key(reg.header)

    def dominates(a: int, b: int) -> bool:
        cur = int(b)
        seen: Set[int] = set()
        while 0 <= cur < len(idom) and cur not in seen:
            if cur == a:
                return True
            seen.add(cur)
            nxt = int(idom[cur])
            if nxt == cur:
                break
            cur = nxt
        return False

    def natural_loop_nodes(back_src: int, header: int) -> Set[int]:
        nodes: Set[int] = {int(header)}
        stack: List[int] = [int(back_src)]
        while stack:
            n = stack.pop()
            if n in nodes:
                continue
            nodes.add(n)
            for p in preds.get(n, []):
                if p not in nodes:
                    stack.append(p)
        return nodes

    loop_headers: Set[int] = {
        bb for bb, lab in enumerate(labels)
        if lab == int(PatternType.LOOP_HEADER)
    }

    def collect_until_merge(
        roots: List[int],
        merge: int,
        stop_nodes: Set[int],
        *,
        stop_loop_headers: Optional[Set[int]] = None,
    ) -> Set[int]:
        stop_loop_headers = stop_loop_headers or set()
        nodes: Set[int] = set()
        stack: List[int] = list(roots)
        while stack:
            n = int(stack.pop())
            if n == merge or n in stop_nodes or n in nodes:
                continue
            if n in stop_loop_headers:
                continue
            nodes.add(n)
            for s in succs.get(n, []):
                if s != merge and s not in stop_nodes:
                    stack.append(s)
        return nodes

    def reaches_merge(
        root_bb: int,
        merge: int,
        stop_nodes: Set[int],
        *,
        stop_loop_headers: Optional[Set[int]] = None,
    ) -> bool:
        """Return whether ``root_bb`` can reach ``merge`` without crossing stops."""
        if merge < 0:
            return False
        stop_loop_headers = stop_loop_headers or set()
        stack: List[int] = [int(root_bb)]
        seen: Set[int] = set()
        while stack:
            n = int(stack.pop())
            if n == merge:
                return True
            if n in stop_nodes or n in seen or n in stop_loop_headers:
                continue
            seen.add(n)
            for s in succs.get(n, []):
                stack.append(int(s))
        return False

    def reaches_target(
        root_bb: int,
        target: int,
        stop_nodes: Set[int],
        *,
        stop_loop_headers: Optional[Set[int]] = None,
    ) -> bool:
        """Return whether ``root_bb`` can reach ``target`` without crossing stops."""
        if target < 0:
            return False
        stop_loop_headers = stop_loop_headers or set()
        stack: List[int] = [int(root_bb)]
        seen: Set[int] = set()
        while stack:
            n = int(stack.pop())
            if n == target:
                return True
            if n in stop_nodes or n in seen or n in stop_loop_headers:
                continue
            seen.add(n)
            for s in succs.get(n, []):
                stack.append(int(s))
        return False

    def branch_contains_later_if_header(
        header: int,
        branch_nodes: Set[int],
        *,
        margin: int = 0,
    ) -> bool:
        """Whether a branch looks like an if/else-if continuation chain."""
        for bb in branch_nodes:
            if int(bb) <= int(header) + int(margin):
                continue
            if labels[int(bb)] == int(PatternType.IF_HEADER):
                return True
        return False

    def branch_contains_structured_body(branch_nodes: Set[int]) -> bool:
        """Whether a branch contains loop/switch evidence and should stay nested."""
        for bb in branch_nodes:
            lab = labels[int(bb)]
            if lab in {
                int(PatternType.LOOP_HEADER),
                int(PatternType.SWITCH_HEADER),
                int(PatternType.LOOP_BODY),
                int(PatternType.SWITCH_CASE_BODY),
            }:
                return True
            for succ in succs.get(int(bb), []):
                if int(succ) <= int(bb):
                    return True
        return False

    def branch_exits_function(root_bb: int, stop_nodes: Set[int]) -> bool:
        """Heuristic for return/error guard branches: reaches a CFG exit before stop."""
        stack: List[int] = [int(root_bb)]
        seen: Set[int] = set()
        while stack:
            n = int(stack.pop())
            if n in stop_nodes or n in seen:
                continue
            seen.add(n)
            ss = succs.get(n, [])
            if not ss:
                return True
            for s in ss:
                stack.append(int(s))
        return False

    def small_branch_body(nodes: Set[int], *, max_nodes: int = 3) -> bool:
        if not nodes:
            return True
        bodyish = [
            bb for bb in nodes
            if labels[int(bb)] not in {
                int(PatternType.IF_HEADER),
                int(PatternType.LOOP_HEADER),
                int(PatternType.SWITCH_HEADER),
            }
        ]
        return len(bodyish) <= max_nodes

    regions_by_header: Dict[int, _RegionInfo] = {}

    # Natural loops first: they define the largest cyclic regions and prevent
    # branch collection from swallowing loop back-edges as straight-line code.
    if edge_index.numel() > 0:
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            u_i, v_i = int(u), int(v)
            if (
                v_i != u_i
                and 0 <= v_i < num_nodes
                and labels[v_i] == int(PatternType.LOOP_HEADER)
                and dominates(v_i, u_i)
            ):
                nodes = natural_loop_nodes(u_i, v_i)
                info = regions_by_header.get(v_i)
                if info is None:
                    regions_by_header[v_i] = _RegionInfo(
                        header=v_i,
                        type="while_loop",
                        pattern_label=int(PatternType.LOOP_HEADER),
                        condition_bbs=[v_i],
                        nodes=set(nodes),
                        merge=-1,
                    )
                else:
                    info.nodes.update(nodes)

    def branch_label_for_nodes(root_bb: int, nodes: Set[int], root_rank: int, n_roots: int) -> int:
        root_lab = labels[root_bb] if 0 <= root_bb < len(labels) else int(PatternType.SEQUENTIAL)
        if root_lab in {int(PatternType.IF_THEN_BODY), int(PatternType.IF_ELSE_BODY)}:
            return int(root_lab)
        # A branch that starts with another IF_HEADER is a nested/continuation
        # control region.  Its descendants' THEN/ELSE labels belong to that
        # child if, not to the current parent.  Fall back to root order rather
        # than letting nested labels make every parent branch look like THEN.
        if n_roots >= 2 and root_lab in {
            int(PatternType.IF_HEADER),
            int(PatternType.LOOP_HEADER),
            int(PatternType.SWITCH_HEADER),
        }:
            return int(PatternType.IF_ELSE_BODY) if root_rank > 0 else int(PatternType.IF_THEN_BODY)
        counts = {
            int(PatternType.IF_THEN_BODY): 0,
            int(PatternType.IF_ELSE_BODY): 0,
        }
        for bb in nodes | {root_bb}:
            lab = labels[bb]
            if lab in counts:
                counts[lab] += 1
        if counts[int(PatternType.IF_ELSE_BODY)] > counts[int(PatternType.IF_THEN_BODY)]:
            return int(PatternType.IF_ELSE_BODY)
        if counts[int(PatternType.IF_THEN_BODY)] > 0:
            return int(PatternType.IF_THEN_BODY)
        if n_roots >= 2 and root_rank > 0:
            return int(PatternType.IF_ELSE_BODY)
        return int(PatternType.IF_THEN_BODY)

    for h, lab in enumerate(labels):
        if lab == int(PatternType.IF_HEADER):
            merge = ipdom[h] if h < len(ipdom) else -1
            roots = [s for s in succs.get(h, []) if s != merge]
            nonempty_branches: List[Tuple[int, Set[int]]] = []
            body: Set[int] = set()
            then_nodes: Set[int] = set()
            else_nodes: Set[int] = set()
            if merge >= 0:
                # If this if-header is inside a loop, one branch may be the
                # loop backedge to the enclosing loop header.  Do not let the
                # if-region swallow that outer loop; a real nested loop inside
                # the if is not stopped because it will not dominate h.
                enclosing_loop_headers = {
                    lh for lh in loop_headers
                    if lh != h and dominates(lh, h)
                }
                branch_infos: List[Tuple[int, int, Set[int], bool]] = []
                ordered_roots = sorted(roots, key=order_key)
                for rank, r in enumerate(ordered_roots):
                    branch_nodes = collect_until_merge(
                        [r],
                        merge,
                        {h},
                        stop_loop_headers=enclosing_loop_headers,
                    )
                    reaches = reaches_merge(
                        r,
                        merge,
                        {h},
                        stop_loop_headers=enclosing_loop_headers,
                    )
                    branch_infos.append((rank, r, branch_nodes, reaches))

                # If one branch reaches the post-dominator and another exits
                # before it, the source-level shape is usually
                # ``if (cond) return;`` followed by the continuation.  Treat the
                # reaching branch as continuation code, not as a child of this
                # if; otherwise early-return chains become ``if { return; if
                # (...) ... }`` and use-before-declaration cascades.
                reaches_flags = [reaches for _, _, nodes_i, reaches in branch_infos if nodes_i]
                prune_continuation = (
                    len(branch_infos) >= 2
                    and any(reaches_flags)
                    and any(not x for x in reaches_flags)
                )
                continuation_roots: Set[int] = set()
                if len(branch_infos) >= 2:
                    continuation_candidates = [
                        (rank, r, branch_nodes, reaches)
                        for rank, r, branch_nodes, reaches in branch_infos
                        if branch_contains_later_if_header(h, branch_nodes)
                        and not branch_contains_structured_body(branch_nodes)
                    ]
                    body_candidates = [
                        (rank, r, branch_nodes, reaches)
                        for rank, r, branch_nodes, reaches in branch_infos
                        if not branch_contains_later_if_header(h, branch_nodes)
                        or branch_contains_structured_body(branch_nodes)
                    ]
                    if continuation_candidates and body_candidates:
                        # Guard-if and else-if lowering often has a short/error
                        # branch and a continuation branch containing later
                        # control headers.  The continuation belongs after this
                        # if, not inside its else body.
                        has_exit_guard = any(
                            branch_exits_function(r, {h, merge})
                            for _rank, r, _nodes, _reaches in body_candidates
                        )
                        has_small_guard = any(
                            small_branch_body(nodes_i) and not branch_contains_structured_body(nodes_i)
                            for _rank, _r, nodes_i, _reaches in body_candidates
                        )
                        if has_exit_guard or has_small_guard:
                            continuation_roots.update(int(r) for _rank, r, _nodes, _reaches in continuation_candidates)
                            stats["n_pruned_if_continuation_branches"] += len(continuation_roots)
                        else:
                            stats["n_protected_structured_branches"] += len(continuation_candidates)

                active_infos = [
                    (rank, r, branch_nodes, reaches)
                    for rank, r, branch_nodes, reaches in branch_infos
                    if not (prune_continuation and reaches)
                    and int(r) not in continuation_roots
                ]
                branch_overrides: Dict[int, int] = {}
                if len(active_infos) >= 2:
                    prelim = {
                        r: branch_label_for_nodes(r, branch_nodes, rank, len(ordered_roots))
                        for rank, r, branch_nodes, _reaches in active_infos
                    }
                    has_then = any(v == int(PatternType.IF_THEN_BODY) for v in prelim.values())
                    has_else = any(v == int(PatternType.IF_ELSE_BODY) for v in prelim.values())
                    if has_then and has_else:
                        branch_overrides.update(prelim)
                    else:
                        # A real binary branch must not serialize both
                        # successors into the same C branch.  This happens when
                        # nested labels from a child if make both roots look
                        # like THEN (or both like ELSE).  Force a two-way split
                        # using CFG successor order; this preserves CFG shape
                        # and lets the condition target/inversion be handled by
                        # later slot filling or repair.
                        cfg_rank = {int(r0): i for i, r0 in enumerate(roots)}
                        active_roots = [int(r) for _rank, r, _nodes, _reaches in active_infos]
                        then_candidates = [
                            r for r in active_roots
                            if labels[r] == int(PatternType.IF_THEN_BODY)
                        ]
                        body_candidates = [
                            r for r in active_roots
                            if labels[r] in {
                                int(PatternType.IF_THEN_BODY),
                                int(PatternType.IF_ELSE_BODY),
                                int(PatternType.LOOP_EXIT),
                                int(PatternType.FUNC_PROLOGUE_EPILOGUE),
                            }
                        ]
                        if then_candidates:
                            then_root = min(then_candidates, key=lambda x: cfg_rank.get(x, x))
                        elif body_candidates:
                            then_root = min(body_candidates, key=lambda x: cfg_rank.get(x, x))
                        else:
                            then_root = min(active_roots, key=lambda x: cfg_rank.get(x, x))
                        for r in active_roots:
                            branch_overrides[r] = (
                                int(PatternType.IF_THEN_BODY)
                                if r == then_root
                                else int(PatternType.IF_ELSE_BODY)
                            )

                for rank, r, branch_nodes, reaches in branch_infos:
                    if prune_continuation and reaches:
                        continue
                    if int(r) in continuation_roots:
                        continue
                    if branch_nodes:
                        nonempty_branches.append((r, branch_nodes))
                    body.update(branch_nodes)
                    branch = branch_overrides.get(
                        int(r),
                        branch_label_for_nodes(r, branch_nodes, rank, len(ordered_roots)),
                    )
                    if branch == int(PatternType.IF_ELSE_BODY):
                        else_nodes.update(branch_nodes)
                    else:
                        then_nodes.update(branch_nodes)
                if len(nonempty_branches) == 1:
                    # With only one non-merge branch, C source has an if-without-
                    # else shape.  The taken branch should serialize as ``then``
                    # even if the per-BB label calls it IF_ELSE_BODY.
                    then_nodes.update(body)
                    else_nodes.clear()
            else:
                # No real join (e.g. one branch returns).  Keep this deliberately
                # narrow so a missing post-dominator does not nest the function tail.
                body_roots = list(roots)
                if len(roots) >= 2:
                    # In early-return chains one successor is the immediate
                    # branch body (usually IF_THEN/IF_ELSE/FUNC_EPILOGUE) while
                    # another successor is the continuation control header.  A
                    # missing real post-dominator used to make us nest that
                    # continuation under this if.  Prefer branch-body-looking
                    # roots when present; if both branches are true bodies
                    # (e.g. both return), both are kept.
                    body_like = [
                        r for r in roots
                        if labels[r] in {
                            int(PatternType.IF_THEN_BODY),
                            int(PatternType.IF_ELSE_BODY),
                            int(PatternType.LOOP_EXIT),
                            int(PatternType.FUNC_PROLOGUE_EPILOGUE),
                        }
                    ]
                    if body_like:
                        body_roots = body_like
                body.update(body_roots)
                then_nodes.update(body_roots)
            regions_by_header[h] = _RegionInfo(
                header=h,
                type="if_else",
                pattern_label=int(PatternType.IF_HEADER),
                condition_bbs=[h],
                nodes={h} | body,
                merge=merge,
                then_nodes=then_nodes,
                else_nodes=else_nodes,
            )
        elif lab == int(PatternType.SWITCH_HEADER):
            merge = ipdom[h] if h < len(ipdom) else -1
            roots = [s for s in succs.get(h, []) if s != merge]
            body = collect_until_merge(roots, merge, {h}) if merge >= 0 else set(roots)
            regions_by_header[h] = _RegionInfo(
                header=h,
                type="switch",
                pattern_label=int(PatternType.SWITCH_HEADER),
                condition_bbs=[h],
                nodes={h} | body,
                merge=merge,
            )

    def refine_regions_with_alignment() -> None:
        """Use M-derived source columns as conservative region hints.

        This stays within the §6.5.1 input contract: only ``M``, pattern labels,
        and the binary CFG are consumed.  Crucially, alignment must not invent
        or delete source-level controls.  Earlier versions collapsed every pair
        of headers with the same source column; that helped a few short-circuit
        cases but frequently erased real sibling/nested ``if`` statements.  The
        refinement below only adjusts branch ownership for nodes that already
        belong to a CFG-derived region.
        """
        if not any(c >= 0 for c in align_col):
            return

        # Reassign ambiguous if branch nodes by M-order when CFG gave a weak
        # then/else split.  This is intentionally local: no region is deleted,
        # and nested control regions remain intact as single ownership units.
        for reg in list(regions_by_header.values()):
            if reg.type != "if_else":
                continue
            roots = [s for s in succs.get(reg.header, []) if s != reg.merge]
            if len(roots) < 2:
                continue
            active_roots = [
                int(r)
                for r in roots
                if r in reg.nodes
                or reaches_target(int(r), reg.header, {reg.header, reg.merge})
                or reaches_merge(int(r), reg.merge, {reg.header})
            ]
            if len(active_roots) < 2:
                continue
            root_cols = [(r, source_col(r)) for r in active_roots if source_col(r) >= 0]
            if len(root_cols) < 2:
                continue
            unique_cols = sorted(set(c for _r, c in root_cols))
            if len(unique_cols) < 2:
                continue

            then_anchor = min(root_cols, key=lambda x: (x[1], order_key(x[0])))[0]
            else_anchor = max(root_cols, key=lambda x: (x[1], order_key(x[0])))[0]
            if then_anchor == else_anchor:
                continue

            movable = (set(reg.then_nodes) | set(reg.else_nodes)) - {reg.header}
            if not movable:
                continue

            # Keep a nested region together under its header.  Moving just a
            # child body without its header is worse than leaving CFG ownership.
            nested_owner: Dict[int, int] = {}
            for child in regions_by_header.values():
                if child.header == reg.header or child.header not in movable:
                    continue
                if child.header in reg.nodes:
                    for bb in child.nodes:
                        if bb in movable:
                            nested_owner[bb] = child.header

            new_then = set(reg.then_nodes)
            new_else = set(reg.else_nodes)
            moved = 0
            visited_nested: Set[int] = set()
            for bb in sorted(movable, key=order_key):
                owner = nested_owner.get(bb)
                if owner is not None:
                    if owner in visited_nested:
                        continue
                    visited_nested.add(owner)
                    group = {
                        x for x, own in nested_owner.items()
                        if own == owner
                    } | {owner}
                    rep = owner
                else:
                    group = {bb}
                    rep = bb
                dt = source_col_distance(rep, then_anchor)
                de = source_col_distance(rep, else_anchor)
                if dt is None or de is None or dt == de:
                    continue
                target_then = dt < de
                before_then = bool(group & new_then)
                before_else = bool(group & new_else)
                if target_then and before_else:
                    new_else.difference_update(group)
                    new_then.update(group)
                    moved += len(group)
                elif (not target_then) and before_then:
                    new_then.difference_update(group)
                    new_else.update(group)
                    moved += len(group)
            if moved:
                reg.then_nodes = new_then & reg.nodes
                reg.else_nodes = new_else & reg.nodes
                stats["n_alignment_refined_if_branches"] += moved

        # Count cases where source alignment suggests more control statements
        # than the available hard pattern headers can express.  This is only a
        # diagnostic; the builder must not synthesize headers from M alone.
        header_cols = {
            source_col(bb)
            for bb, lab in enumerate(labels)
            if lab in {
                int(PatternType.IF_HEADER),
                int(PatternType.LOOP_HEADER),
                int(PatternType.SWITCH_HEADER),
            }
            and source_col(bb) >= 0
        }
        branch_like_cols = {
            source_col(bb)
            for bb, lab in enumerate(labels)
            if lab in {
                int(PatternType.IF_THEN_BODY),
                int(PatternType.IF_ELSE_BODY),
                int(PatternType.LOOP_BODY),
                int(PatternType.SWITCH_CASE_BODY),
            }
            and source_col(bb) >= 0
        }
        missing_like = branch_like_cols - header_cols
        if missing_like:
            stats["n_alignment_header_insufficient"] += len(missing_like)

    refine_regions_with_alignment()

    def collapse_if_header_chains() -> None:
        """Merge short-circuit IF_HEADER chains into one source-level if.

        A C condition such as ``a && b && c`` often appears in the binary as a
        chain of conditional BBs.  Module-A labels each as IF_HEADER, but the C
        skeleton should contain one ``if`` condition slot aligned to all header
        BBs.  If header ``h`` only routes into child IF_HEADER ``r`` before the
        same post-dominator merge, collapse ``r`` into ``h`` and keep ``r``'s
        real branch bodies.
        """

        def branch_reaches_parent_merge(child: _RegionInfo, parent: _RegionInfo) -> bool:
            if parent.merge < 0:
                return False
            for s in succs.get(child.header, []):
                if int(s) == parent.merge:
                    return True
                if reaches_merge(int(s), parent.merge, {parent.header, child.header}):
                    return True
            return False

        def child_has_independent_body(child: _RegionInfo) -> bool:
            # An explicit else region is independent source structure.  A single
            # then-side statement may still be the final body of a short-circuit
            # condition, so do not use mere body presence as a hard veto.
            if child.else_nodes:
                return True
            nested_controls = [
                r for r in regions_by_header.values()
                if r.header != child.header and r.header in child.nodes
            ]
            if any(r.type in {"while_loop", "for_loop", "switch"} for r in nested_controls):
                return True
            # Do not reject merely because the child contains further IFs:
            # optimized short-circuit expressions often lower to a chain of
            # IF_HEADER regions ending in one real body.  The collapsed parent
            # keeps the child's final then/else body after condition BBs are
            # merged, so this remains source-free and region-preserving.
            return False

        def alignment_vetoes_condition_collapse(parent: _RegionInfo, child: _RegionInfo) -> bool:
            pc, cc = source_col(parent.header), source_col(child.header)
            if pc < 0 or cc < 0:
                return False
            if pc == cc:
                return False
            child_body = (child.then_nodes | child.else_nodes) - set(child.condition_bbs)
            if not child_body:
                return False
            body_cols = source_col_set(child_body)
            if not body_cols:
                return False
            # Different source columns plus body evidence means the child is
            # probably an independent source-level if, not just another lowered
            # condition BB.
            return any(c != pc for c in body_cols)

        changed = True
        while changed:
            changed = False
            for h in sorted(list(regions_by_header)):
                reg = regions_by_header.get(h)
                if reg is None or reg.type != "if_else" or reg.merge < 0:
                    continue
                roots = [s for s in succs.get(h, []) if s != reg.merge]
                child_roots = [
                    r for r in roots
                    if r in regions_by_header
                    and regions_by_header[r].type == "if_else"
                ]
                if len(child_roots) != 1:
                    continue
                child = regions_by_header[child_roots[0]]
                same_merge = child.merge >= 0 and child.merge == reg.merge
                if not same_merge and not branch_reaches_parent_merge(child, reg):
                    continue
                if alignment_vetoes_condition_collapse(reg, child):
                    continue
                if child_has_independent_body(child):
                    continue
                non_child_body = (reg.nodes - {reg.header}) - child.nodes
                if non_child_body:
                    continue
                reg.condition_bbs = sorted(
                    set(reg.condition_bbs) | set(child.condition_bbs),
                    key=order_key,
                )
                reg.nodes = {reg.header} | set(child.nodes)
                reg.then_nodes = set(child.then_nodes)
                reg.else_nodes = set(child.else_nodes)
                del regions_by_header[child.header]
                stats["n_collapsed_if_header_chains"] += 1
                changed = True
                break

    collapse_if_header_chains()

    def flatten_degenerate_if_else_chains() -> None:
        """Lift source-like sequential guard-if chains out of synthetic else nesting.

        Optimized binaries often encode a run of source-level sibling guard
        checks as ``if (...) ... else if (...) ...`` in raw CFG shape.  If a
        parent if's else region consists only of a child if and small statement
        bodies, prefer sibling structure.  This is deliberately source-free and
        preserves the child region as an independent CST node.
        """
        changed = True
        while changed:
            changed = False
            for h in sorted(list(regions_by_header)):
                reg = regions_by_header.get(h)
                if reg is None or reg.type != "if_else" or not reg.else_nodes:
                    continue
                child_headers = [
                    r.header for r in regions_by_header.values()
                    if r.header != h and r.type == "if_else" and r.header in reg.else_nodes
                ]
                if len(child_headers) != 1:
                    continue
                child = regions_by_header[child_headers[0]]
                leftover_else = set(reg.else_nodes) - set(child.nodes)
                leftover_else.discard(child.header)
                if not small_branch_body(leftover_else, max_nodes=2):
                    continue
                if branch_contains_structured_body(child.nodes) or branch_contains_structured_body(leftover_else):
                    stats["n_protected_structured_branches"] += 1
                    continue
                # Keep a genuine nested else body when the child is fully within
                # an exclusive non-continuation branch.  Otherwise lift the child
                # by removing it from the parent's region and clearing the else.
                reg.nodes.difference_update(child.nodes)
                reg.else_nodes.clear()
                stats["n_flattened_if_else_chains"] += 1
                changed = True
                break

    flatten_degenerate_if_else_chains()

    regions: List[_RegionInfo] = list(regions_by_header.values())

    parent_of: Dict[int, Optional[int]] = {}
    for reg in regions:
        parent: Optional[int] = None
        parent_size: Optional[int] = None
        parent_align_score: Optional[Tuple[int, int, float]] = None
        for cand in regions:
            if cand.header == reg.header:
                continue
            if reg.header not in cand.nodes:
                continue
            if cand.type == "if_else":
                if reg.header not in cand.then_nodes and reg.header not in cand.else_nodes:
                    continue
                cand_roots = [s for s in succs.get(cand.header, []) if s != cand.merge]
                reaching_roots = [
                    r for r in cand_roots
                    if reaches_target(r, reg.header, {cand.header, cand.merge})
                ]
                if len(reaching_roots) != 1:
                    stats["n_lifted_nonexclusive_if_children"] += 1
                    continue
            if len(cand.nodes) <= len(reg.nodes):
                continue
            cand_size = len(cand.nodes)
            cand_range = source_col_range(cand.nodes)
            reg_range = source_col_range(reg.nodes)
            if cand_range is not None and reg_range is not None:
                contains = int(cand_range[0] <= reg_range[0] and reg_range[1] <= cand_range[1])
                header_dist = source_col_distance(reg.header, cand.header)
                if header_dist is None:
                    header_dist = 10**9
                cand_center = source_col_center(cand.nodes)
                reg_center = source_col_center(reg.nodes)
                center_dist = (
                    abs(float(reg_center) - float(cand_center))
                    if cand_center is not None and reg_center is not None
                    else float("inf")
                )
                align_score = (1 - contains, int(header_dist), float(center_dist))
            else:
                align_score = (1, 10**9, float("inf"))

            choose = False
            if parent_size is None or cand_size < parent_size:
                choose = True
            elif cand_size == parent_size and parent_align_score is not None and align_score < parent_align_score:
                choose = True
                stats["n_alignment_parent_tiebreaks"] += 1
            if choose:
                parent = cand.header
                parent_size = cand_size
                parent_align_score = align_score
        parent_of[reg.header] = parent

    children_by_parent: Dict[Optional[int], List[_RegionInfo]] = {}
    for reg in regions:
        children_by_parent.setdefault(parent_of.get(reg.header), []).append(reg)
    for regs in children_by_parent.values():
        regs.sort(key=region_order_key)

    def branch_for(parent: Optional[_RegionInfo], bb: int) -> Optional[int]:
        if parent is None or parent.type != "if_else":
            return None
        if bb in parent.else_nodes:
            return int(PatternType.IF_ELSE_BODY)
        if bb in parent.then_nodes:
            return int(PatternType.IF_THEN_BODY)
        return None

    def build_region_node(reg: _RegionInfo) -> CSTNode:
        node = CSTNode(
            type=reg.type,
            condition_bb=reg.condition_bbs[0] if reg.condition_bbs else reg.header,
            aligned_bbs=sorted(reg.nodes, key=order_key),
            pattern_label=reg.pattern_label,
        )
        if reg.type in {"if_else", "while_loop", "for_loop", "switch"}:
            node.aligned_bbs = list(reg.condition_bbs or [reg.header])
        node.children = build_children(reg)
        return node

    def build_children(parent: Optional[_RegionInfo]) -> List[CSTNode]:
        parent_header = parent.header if parent is not None else None
        parent_nodes = set(parent.nodes) if parent is not None else set(range(num_nodes))
        nested_regions = children_by_parent.get(parent_header, [])
        nested_nodes: Set[int] = set()
        for reg in nested_regions:
            nested_nodes.update(reg.nodes)

        events: List[Tuple[Tuple[float, float, int], int, CSTNode]] = []
        for reg in nested_regions:
            child = build_region_node(reg)
            child.branch_label = branch_for(parent, reg.header)
            events.append((region_order_key(reg), 0, child))

        for bb in parent_nodes:
            if parent is not None and bb == parent.header:
                continue
            if bb in nested_nodes:
                continue
            # A header label that did not produce a structured region is usually
            # a compiler routing block or a noisy pattern prediction.  Emitting it
            # as a normal statement gives the repacker a chance to paste an entire
            # source body before the real control construct.
            if labels[bb] in {
                int(PatternType.IF_HEADER),
                int(PatternType.LOOP_HEADER),
                int(PatternType.SWITCH_HEADER),
            }:
                continue
            child = CSTNode(
                type="statement",
                aligned_bbs=[bb],
                pattern_label=labels[bb],
                branch_label=branch_for(parent, bb),
            )
            events.append((order_key(bb), 1, child))

        events.sort(key=lambda x: (x[0], x[1]))
        return [node for _, _, node in events]

    root.children = build_children(None)

    def has_compatible_parent(bb: int, compatible_headers: Set[int]) -> bool:
        for reg in regions:
            if reg.pattern_label in compatible_headers and bb in reg.nodes and bb != reg.header:
                return True
        return False

    for bb, lab in enumerate(labels):
        if lab in {int(PatternType.IF_THEN_BODY), int(PatternType.IF_ELSE_BODY)}:
            if not has_compatible_parent(bb, {int(PatternType.IF_HEADER)}):
                stats["n_body_unattached"] += 1
        elif lab in {int(PatternType.LOOP_BODY), int(PatternType.LOOP_EXIT)}:
            if not has_compatible_parent(bb, {int(PatternType.LOOP_HEADER)}):
                stats["n_loop_exit_orphan"] += 1
        elif lab == int(PatternType.SWITCH_CASE_BODY):
            if not has_compatible_parent(bb, {int(PatternType.SWITCH_HEADER)}):
                stats["n_body_unattached"] += 1

    return root, stats


# ═══════════════════════ Skeleton reliability score ══════════════════════════

def compute_pattern_reliability(
    pattern_probs: torch.Tensor,
    pattern_labels: torch.Tensor,
    stats: Optional[Dict[str, int]] = None,
    *,
    header_weight: float = 2.0,
    tree_weight: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """§6.5 推理时的骨架可靠性分数 ``R ∈ [0, 1]``。

    分两部分：

    * ``s_pat``: 每个 BB 上 ``max_c p_{i,c}`` 的（按 header 加权的）几何平均；
      header BB（LOOP/IF/SWITCH_HEADER）权重为 ``header_weight``，普通 BB 为 1。
      头预测一旦发飘（softmax 变平），``s_pat`` 显著下降。
    * ``s_tree``: ``1 - (n_body_unattached + n_loop_exit_orphan) / n_nodes``；
      在 :func:`build_control_structure_tree_with_stats` 的 ``stats`` 上算。

    最终 ``R = sqrt(s_pat * s_tree ** tree_weight)``。``stats=None`` 时仅取 ``s_pat``。

    Parameters
    ----------
    pattern_probs : Tensor [n_b, C]
        Module A 的 ``softmax(pattern_logits)``（或等价 one-hot；one-hot 时 R≈1）。
    pattern_labels : Tensor [n_b]
        每个 BB 的离散模式标签（一般为 ``pattern_probs.argmax``）。
    stats : dict, optional
        建树过程统计（见 :func:`build_control_structure_tree_with_stats`）。
    header_weight : float
        header BB 在几何平均里的相对权重。
    tree_weight : float
        ``s_tree`` 进入合成时的指数（``0`` 即忽略建树自检）。
    """
    if pattern_probs.numel() == 0:
        return 1.0
    n_b = int(pattern_probs.size(0))
    if n_b == 0:
        return 1.0

    probs = pattern_probs.detach().to(torch.float32)
    max_p = probs.max(dim=-1).values  # [n_b]
    labels = pattern_labels.detach().to(torch.long)

    weights = torch.ones(n_b, device=probs.device, dtype=torch.float32)
    if header_weight != 1.0:
        is_header = (
            (labels == int(PatternType.LOOP_HEADER))
            | (labels == int(PatternType.IF_HEADER))
            | (labels == int(PatternType.SWITCH_HEADER))
        )
        weights = torch.where(
            is_header,
            torch.full_like(weights, float(header_weight)),
            weights,
        )

    log_p = torch.log(max_p.clamp(min=eps))
    s_pat = float(
        torch.exp((log_p * weights).sum() / weights.sum().clamp(min=eps)).item()
    )
    s_pat = max(0.0, min(1.0, s_pat))

    s_tree = 1.0
    if stats:
        bad = int(stats.get("n_body_unattached", 0)) + int(stats.get("n_loop_exit_orphan", 0))
        denom = max(1, int(stats.get("n_nodes", 1)))
        s_tree = max(0.0, 1.0 - float(bad) / float(denom))

    tree_term = s_tree ** max(0.0, float(tree_weight))
    return float((s_pat * tree_term) ** 0.5)


def build_skeleton_for_binary_cfg(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    block_instructions_text: Optional[Dict[int, str]] = None,
    pattern_probs: Optional[torch.Tensor] = None,
    bb_order_key: Optional[Sequence[object] | Dict[int, object]] = None,
    alignment_matrix: Optional[torch.Tensor] = None,
    predicted_source_count: Optional[float] = None,
    header_prefix_slots: Optional[Dict[int, int]] = None,
    skip_statement_bbs: Optional[Set[int]] = None,
) -> Skeleton:
    """§6.5.1：由 ``argmax`` 模式标签与 CFG 拓扑构建 CST，再序列化为带 FILL 的骨架文本。

    若提供 ``pattern_probs``（Module A 的 ``softmax(pattern_logits)``），同时把
    建树统计 + pattern 分布合成的可靠性分数 ``R`` 写进 :attr:`Skeleton.reliability`，
    并把 stats 写进 :attr:`Skeleton.build_stats`，供 §6.5 后续 soft 用法使用。
    训练数据路径（pattern 来自 GT 标签、无 logits）时保持 ``None``，向后兼容。
    """
    cst, stats = _build_cst_impl(
        pattern_labels,
        edge_index,
        num_nodes,
        bb_order_key,
        alignment_matrix=alignment_matrix,
    )
    text_map: Dict[int, str] = dict(block_instructions_text or {})
    for i in range(num_nodes):
        text_map.setdefault(i, f"BB_{i}")
    header_prefix = dict(header_prefix_slots or {})
    skip_bbs = set(int(x) for x in (skip_statement_bbs or set()))

    if alignment_matrix is not None and alignment_matrix.numel() > 0:
        try:
            M = alignment_matrix.detach().to(torch.float32).cpu()
            if M.dim() == 2 and M.size(0) >= num_nodes and M.size(1) > 0:
                rows = M[:num_nodes]
                labels = [int(x) for x in pattern_labels.detach().cpu().tolist()]
                argmax = rows.argmax(dim=1).tolist()
                mass = rows.sum(dim=1).tolist()
                # Rows that align to the earliest source column but are not
                # headers often correspond to prologue/epilogue/dummy code.
                # They should not become ordinary source statements.
                for bb, lab in enumerate(labels):
                    if lab in {
                        int(PatternType.IF_HEADER),
                        int(PatternType.LOOP_HEADER),
                        int(PatternType.SWITCH_HEADER),
                    }:
                        continue
                    if float(mass[bb]) <= 1e-12:
                        skip_bbs.add(bb)
                    elif int(argmax[bb]) == 0 and bb != 0:
                        skip_bbs.add(bb)
        except Exception:
            pass

    skeleton = serialize_cst_to_skeleton(
        cst,
        text_map,
        header_prefix_slots=header_prefix,
        skip_statement_bbs=skip_bbs,
    )
    skeleton.build_stats = stats
    if predicted_source_count is not None:
        try:
            pred = float(predicted_source_count)
            skeleton.build_stats = dict(skeleton.build_stats or {})
            skeleton.build_stats["predicted_source_count"] = int(round(max(0.0, pred)))
            skeleton.build_stats["n_slots"] = len(skeleton.slots)
            skeleton.build_stats["slot_source_count_gap"] = int(
                len(skeleton.slots) - round(max(0.0, pred))
            )
        except Exception:
            pass
    if pattern_probs is not None and pattern_probs.numel() > 0:
        skeleton.reliability = compute_pattern_reliability(
            pattern_probs, pattern_labels, stats,
        )
    return skeleton


def enrich_skeleton_slots_with_alignment_matrix(
    skeleton: Skeleton,
    M: torch.Tensor,
) -> None:
    """§6.5.2：按软对齐矩阵 ``M`` 为每个 slot 标注主峰源码语句下标（写入 ``binary_context_text``）。"""
    if M.numel() == 0:
        return
    n_b, n_s = M.shape[0], M.shape[1]
    for slot in skeleton.slots:
        if not slot.aligned_bb_indices:
            continue
        idxs = [
            i for i in slot.aligned_bb_indices
            if 0 <= i < n_b
        ]
        if not idxs:
            continue
        ii = torch.tensor(idxs, device=M.device, dtype=torch.long)
        mass = M[ii].sum(dim=0)
        j = int(mass.argmax().item()) if n_s > 0 else -1
        tag = f"aligned_stmt_idx={j}" if j >= 0 else "aligned_stmt_idx=none"
        if slot.binary_context_text:
            slot.binary_context_text = f"{slot.binary_context_text} | {tag}"
        else:
            slot.binary_context_text = tag


def build_bb_opcode_hints_from_batch(
    graph_index: int,
    bin_batch: torch.Tensor,
    bin_block_lengths: torch.Tensor,
    bin_opcodes: torch.Tensor,
    max_opcodes_per_bb: int = 6,
) -> Dict[int, str]:
    """为单个函数样本构造 ``bb_index -> 简短 opcode 摘要``，供 :func:`serialize_cst_to_skeleton` 作 bin_ctx。"""
    if bin_block_lengths.numel() == 0:
        return {}
    g = int(graph_index)
    instr_off = 0
    local_bb = 0
    out: Dict[int, str] = {}
    for i in range(bin_block_lengths.size(0)):
        bl = int(bin_block_lengths[i].item())
        gi = int(bin_batch[i].item())
        if gi == g:
            sl = bin_opcodes[instr_off : instr_off + bl]
            k = min(bl, max_opcodes_per_bb)
            parts = [str(int(x)) for x in sl[:k].tolist()]
            if bl > max_opcodes_per_bb:
                parts.append("...")
            out[local_bb] = f"BB{local_bb} opc[{','.join(parts)}]"
            local_bb += 1
        instr_off += bl
    return out


# ═══════════════════════ CST -> Skeleton ═════════════════════════════════════

_PATTERN_NAME: Dict[int, str] = {
    int(PatternType.SEQUENTIAL): "sequential",
    int(PatternType.IF_HEADER): "if_header",
    int(PatternType.IF_THEN_BODY): "if_then_body",
    int(PatternType.IF_ELSE_BODY): "if_else_body",
    int(PatternType.LOOP_HEADER): "loop_header",
    int(PatternType.LOOP_BODY): "loop_body",
    int(PatternType.LOOP_EXIT): "loop_exit",
    int(PatternType.SWITCH_HEADER): "switch_header",
    int(PatternType.SWITCH_CASE_BODY): "switch_case_body",
    int(PatternType.FUNC_PROLOGUE_EPILOGUE): "func_prologue_epilogue",
}


def serialize_cst_to_skeleton(
    cst: CSTNode,
    block_instructions_text: Dict[int, str],
    *,
    header_prefix_slots: Optional[Dict[int, int]] = None,
    skip_statement_bbs: Optional[Set[int]] = None,
) -> Skeleton:
    """§6.5.2：CST → 带 ``<FILL_*>`` 的 C 骨架文本。

    严格按文档示例格式：

    * ``function``   → ``<FILL_TYPE:ret_type> func(<FILL_PARAM:params>) { ... }``
    * ``while_loop`` → ``while (<FILL_COND:slot_N>) { ... }``
    * ``if_else``    → ``if (<FILL_COND:slot_N>) { ... } else { ... }``
    * FUNC_PROLOGUE_EPILOGUE → ``<FILL_EXPR:slot_N>;`` 或 ``return <FILL_EXPR:slot_N>;``
    * 其余 statement  → ``<FILL_STMT:slot_N>;``
    * 每行尾附 ``// aligned to BB_k (pattern_name)``
    """
    slots: List[SlotInfo] = []
    _slot_counter: List[int] = [0]
    prefix_slots: Dict[int, int] = {
        int(k): max(0, int(v))
        for k, v in (header_prefix_slots or {}).items()
        if int(v) > 0
    }
    skip_bbs: Set[int] = set(int(x) for x in (skip_statement_bbs or set()))

    def _next_slot_id() -> str:
        sid: str = f"slot_{_slot_counter[0]}"
        _slot_counter[0] += 1
        return sid

    def _bb_context(bbs: List[int]) -> str:
        parts: List[str] = []
        for bb in bbs:
            text: str = block_instructions_text.get(bb, "")
            if text:
                parts.append(f"BB{bb}: {text}")
        return " | ".join(parts) if parts else ""

    def _bb_comment(bbs: List[int], pattern_label: Optional[int] = None) -> str:
        """§6.5.2 行尾注释：用块注释避免 ``//`` 吃掉同行闭合大括号。"""
        if not bbs:
            return ""
        bb_str = ", ".join(f"BB_{b}" for b in bbs)
        pname = _PATTERN_NAME.get(int(pattern_label), "") if pattern_label is not None else ""
        if pname:
            return f"/* aligned to {bb_str} ({pname}) */"
        return f"/* aligned to {bb_str} */"

    def _emit_header_prefix_slots(cond_bbs: List[int], indent: int) -> str:
        """Emit FILL_STMT slots for statements that share a header BB.

        A binary header block often contains setup code immediately followed by
        the conditional branch, e.g. ``call locale_charset`` then ``cmp``.  A
        pure ``if (<FILL_COND>)`` slot cannot represent that setup statement,
        so callers may request one or more prefix statement slots aligned to
        the same header BB.
        """
        lines: List[str] = []
        pad = "    " * indent
        for bb in cond_bbs:
            n_prefix = int(prefix_slots.get(int(bb), 0))
            for _ in range(max(0, n_prefix)):
                stmt_slot_id = _next_slot_id()
                bbs = [int(bb)]
                slots.append(SlotInfo(
                    slot_id=stmt_slot_id,
                    slot_type="FILL_STMT",
                    aligned_bb_indices=bbs,
                    binary_context_text=(
                        _bb_context(bbs) + " | role=header_prefix"
                        if _bb_context(bbs)
                        else "role=header_prefix"
                    ),
                ))
                bb_str = ", ".join(f"BB_{b}" for b in bbs)
                comment = f"/* aligned to {bb_str} (header_prefix) */"
                lines.append(f"{pad}<FILL_STMT:{stmt_slot_id}>;  {comment}".rstrip())
        return "\n".join(lines)

    def _is_epilogue_position(node: CSTNode, parent_children: List[CSTNode]) -> bool:
        """判断 FUNC_PROLOGUE_EPILOGUE 节点是否在函数末尾（return 位置）。"""
        if node.pattern_label != int(PatternType.FUNC_PROLOGUE_EPILOGUE):
            return False
        if not parent_children:
            return False
        return node is parent_children[-1]

    def _serialize(
        node: CSTNode,
        indent: int,
        parent_children: Optional[List[CSTNode]] = None,
    ) -> str:
        pad: str = "    " * indent

        if node.type == "function":
            # §6.5.2：函数签名用 FILL_TYPE + FILL_PARAM
            ret_slot = _next_slot_id()
            param_slot = _next_slot_id()
            slots.append(SlotInfo(
                slot_id=ret_slot, slot_type="FILL_TYPE",
                aligned_bb_indices=[], binary_context_text="",
            ))
            slots.append(SlotInfo(
                slot_id=param_slot, slot_type="FILL_PARAM",
                aligned_bb_indices=[], binary_context_text="",
            ))
            body_lines: List[str] = []
            for child in node.children:
                rendered = _serialize(child, indent + 1, node.children)
                if rendered:
                    body_lines.append(rendered)
            body: str = "\n".join(body_lines)
            return (
                f"<FILL_TYPE:{ret_slot}> func(<FILL_PARAM:{param_slot}>) {{\n"
                f"{body}\n}}"
            )

        elif node.type in ("while_loop", "for_loop"):
            kw = "while" if node.type == "while_loop" else "for"
            cond_bbs: List[int] = (
                list(node.aligned_bbs)
                if node.aligned_bbs
                else ([node.condition_bb] if node.condition_bb is not None else [])
            )
            prefix = _emit_header_prefix_slots(cond_bbs, indent)
            cond_slot_id: str = _next_slot_id()
            comment = _bb_comment(cond_bbs, node.pattern_label)
            slots.append(SlotInfo(
                slot_id=cond_slot_id, slot_type="FILL_COND",
                aligned_bb_indices=cond_bbs,
                binary_context_text=_bb_context(cond_bbs),
            ))
            body_lines = []
            for child in node.children:
                rendered = _serialize(child, indent + 1, node.children)
                if rendered:
                    body_lines.append(rendered)
            body = "\n".join(body_lines)
            header = f"{pad}{kw} (<FILL_COND:{cond_slot_id}>)"
            if comment:
                header += f" {{{comment.rjust(0)}"
                result = f"{header}\n{body}\n{pad}}}"
            else:
                result = f"{header} {{\n{body}\n{pad}}}"
            return f"{prefix}\n{result}" if prefix else result

        elif node.type == "if_else":
            cond_bbs = (
                list(node.aligned_bbs)
                if node.aligned_bbs
                else ([node.condition_bb] if node.condition_bb is not None else [])
            )
            prefix = _emit_header_prefix_slots(cond_bbs, indent)
            cond_slot_id = _next_slot_id()
            comment = _bb_comment(cond_bbs, node.pattern_label)
            slots.append(SlotInfo(
                slot_id=cond_slot_id, slot_type="FILL_COND",
                aligned_bb_indices=cond_bbs,
                binary_context_text=_bb_context(cond_bbs),
            ))
            then_children: List[CSTNode] = []
            else_children: List[CSTNode] = []
            for child in node.children:
                child_branch = (
                    child.branch_label
                    if child.branch_label is not None
                    else child.pattern_label
                )
                if child_branch == int(PatternType.IF_ELSE_BODY):
                    else_children.append(child)
                else:
                    then_children.append(child)
            if not then_children and not else_children:
                then_children = list(node.children)

            # Assign slot ids in textual order.  The old code serialized
            # children while classifying them, so an else child visited before a
            # then child could allocate slot_3..slot_7 even though the then body
            # was printed first as slot_8.  That desynchronised debug fields and
            # made SFT fill-token order unnecessarily non-monotonic.
            then_parts = [
                part
                for child in then_children
                for part in [_serialize(child, indent + 1, node.children)]
                if part
            ]
            else_parts = [
                part
                for child in else_children
                for part in [_serialize(child, indent + 1, node.children)]
                if part
            ]

            then_body: str = "\n".join(then_parts) if then_parts else ""
            header_line = f"{pad}if (<FILL_COND:{cond_slot_id}>)"
            if comment:
                header_line += f" {{{comment.rjust(0)}"
            else:
                header_line += " {"
            result: str = f"{header_line}\n{then_body}\n{pad}}}"
            if else_parts:
                else_body: str = "\n".join(else_parts)
                result += f" else {{\n{else_body}\n{pad}}}"
            return f"{prefix}\n{result}" if prefix else result

        elif node.type == "switch":
            cond_bbs = (
                list(node.aligned_bbs)
                if node.aligned_bbs
                else ([node.condition_bb] if node.condition_bb is not None else [])
            )
            prefix = _emit_header_prefix_slots(cond_bbs, indent)
            cond_slot_id = _next_slot_id()
            comment = _bb_comment(cond_bbs, node.pattern_label)
            slots.append(SlotInfo(
                slot_id=cond_slot_id, slot_type="FILL_COND",
                aligned_bb_indices=cond_bbs,
                binary_context_text=_bb_context(cond_bbs),
            ))
            body_lines = []
            for child in node.children:
                rendered = _serialize(child, indent + 1, node.children)
                if rendered:
                    body_lines.append(rendered)
            body = "\n".join(body_lines)
            header_line = f"{pad}switch (<FILL_COND:{cond_slot_id}>)"
            if comment:
                header_line += f" {{{comment.rjust(0)}"
            else:
                header_line += " {"
            result = f"{header_line}\n{body}\n{pad}}}"
            return f"{prefix}\n{result}" if prefix else result

        else:
            # statement 节点：根据 pattern_label 区分 FILL_EXPR / FILL_STMT
            stmt_slot_id: str = _next_slot_id()
            bbs = list(node.aligned_bbs)
            if bbs and all(int(bb) in skip_bbs for bb in bbs):
                return ""
            comment = _bb_comment(bbs, node.pattern_label)

            if node.pattern_label == int(PatternType.FUNC_PROLOGUE_EPILOGUE):
                slot_type = "FILL_EXPR"
                slots.append(SlotInfo(
                    slot_id=stmt_slot_id, slot_type=slot_type,
                    aligned_bb_indices=bbs,
                    binary_context_text=_bb_context(bbs),
                ))
                is_return = _is_epilogue_position(node, parent_children or [])
                if is_return:
                    line = f"{pad}return <FILL_EXPR:{stmt_slot_id}>;"
                else:
                    line = f"{pad}<FILL_EXPR:{stmt_slot_id}>;"
            else:
                slot_type = "FILL_STMT"
                slots.append(SlotInfo(
                    slot_id=stmt_slot_id, slot_type=slot_type,
                    aligned_bb_indices=bbs,
                    binary_context_text=_bb_context(bbs),
                ))
                line = f"{pad}<FILL_STMT:{stmt_slot_id}>;"

            if comment:
                line += f"  {comment}"
            return line

    skeleton_text: str = _serialize(cst, indent=0)
    return Skeleton(skeleton_text=skeleton_text, slots=slots)


# ═══════════════════════ slot attention mask ═════════════════════════════════

def build_slot_mask(
    skeleton: Skeleton,
    num_nodes: int,
) -> torch.Tensor:
    """Build a binary mask indicating which basic blocks each slot attends to.

    The mask is used in Module C's cross-attention layers to restrict each
    ``<FILL_*>`` token to attend only to the graph embeddings of the basic
    blocks that the alignment module assigned to that slot.

    Args:
        skeleton: A ``Skeleton`` instance produced by
                  :func:`serialize_cst_to_skeleton`.
        num_nodes: Total number of basic blocks in the binary CFG.

    Returns:
        mask: Shape ``[num_slots, num_nodes]`` (float32).  ``mask[s, b] = 1``
              if slot *s* should attend to block *b*, else 0.
    """
    num_slots: int = len(skeleton.slots)
    mask: torch.Tensor = torch.zeros(
        num_slots, num_nodes, dtype=torch.float32
    )

    for slot_idx, slot in enumerate(skeleton.slots):
        for bb_idx in slot.aligned_bb_indices:
            if 0 <= bb_idx < num_nodes:
                mask[slot_idx, bb_idx] = 1.0

    return mask
