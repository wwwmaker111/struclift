"""
Graph-theoretic utilities for StrucLift binary CFG analysis.

Provides adjacency matrix construction, dominator tree computation,
loop header detection, structural feature extraction, and region
segmentation from pattern labels.

Uses NetworkX for dominator tree and loop detection (battle-tested algorithms).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import torch

from struclift.data.structures import (
    ASTType,
    BasicBlock,
    BinaryCFG,
    EdgeType,
    PatternType,
)


def _edge_index_to_nx_digraph(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> nx.DiGraph:
    """Convert edge_index tensor to NetworkX DiGraph."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        G.add_edges_from(zip(src_list, dst_list))
    return G


# ═══════════════════════ adjacency construction ══════════════════════════════

def build_adjacency_matrix(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_edge_types: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build dense adjacency matrices from edge list.

    Args:
        edge_index: Edge endpoints.  Shape ``[2, num_edges]`` (int64).
                    ``edge_index[0]`` = source, ``edge_index[1]`` = target.
        edge_type:  Per-edge type label.  Shape ``[num_edges]`` (int64).
        num_nodes:  Number of nodes in the graph.
        num_edge_types: Number of distinct edge types
                        (default 5: COND, UNCOND, BACK, CALL, RET).

    Returns:
        adj:       Untyped adjacency (binary).
                   Shape ``[num_nodes, num_nodes]`` (float32).
        typed_adj: Per-type adjacency.
                   Shape ``[num_edge_types, num_nodes, num_nodes]`` (float32).
    """
    device: torch.device = edge_index.device

    adj: torch.Tensor = torch.zeros(
        num_nodes, num_nodes, dtype=torch.float32, device=device
    )
    typed_adj: torch.Tensor = torch.zeros(
        num_edge_types, num_nodes, num_nodes, dtype=torch.float32, device=device
    )

    if edge_index.numel() == 0:
        return adj, typed_adj

    src: torch.Tensor = edge_index[0]  # [num_edges]
    dst: torch.Tensor = edge_index[1]  # [num_edges]
    etype: torch.Tensor = edge_type.long()  # [num_edges]

    adj[src, dst] = 1.0

    for t in range(num_edge_types):
        mask: torch.Tensor = (etype == t)
        if mask.any():
            typed_adj[t, src[mask], dst[mask]] = 1.0

    return adj, typed_adj


# ═══════════════════════ dominator tree ═══════════════════════════════════════

def compute_dominator_tree(
    edge_index: torch.Tensor,
    num_nodes: int,
    entry_node: int = 0,
) -> torch.Tensor:
    """Compute the immediate dominator tree using NetworkX.

    Uses ``nx.immediate_dominators`` (Cooper-Harvey-Kennedy / Lengauer-Tarjan).

    Args:
        edge_index: Edge endpoints.  Shape ``[2, num_edges]`` (int64).
        num_nodes:  Number of nodes.
        entry_node: Index of the CFG entry node (default 0).

    Returns:
        parent: Immediate dominator for each node.
                Shape ``[num_nodes]`` (int64).
                ``parent[entry_node] == entry_node`` (self-loop at root).
                Unreachable nodes have ``parent[i] == -1``.
    """
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.int64)

    G = _edge_index_to_nx_digraph(edge_index, num_nodes)

    try:
        idom_dict = nx.immediate_dominators(G, entry_node)
    except nx.NetworkXError:
        # entry_node not in G or other error: fallback to all unreachable
        return torch.full((num_nodes,), -1, dtype=torch.int64)

    # Convert to list: idom[entry] = entry; idom[node] = idom_dict.get(node, -1)
    idom: List[int] = [-1] * num_nodes
    idom[entry_node] = entry_node
    for node, dom in idom_dict.items():
        if 0 <= node < num_nodes:
            idom[node] = dom

    return torch.tensor(idom, dtype=torch.int64)


# ═══════════════════════ loop header detection ═══════════════════════════════

def find_loop_headers(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Set[int]:
    """Identify loop headers by detecting back edges (NetworkX-based).

    A back edge is an edge ``(u, v)`` where ``v`` dominates ``u``.
    The target ``v`` of every back edge is a loop header.

    Args:
        edge_index: Edge endpoints.  Shape ``[2, num_edges]`` (int64).
        num_nodes:  Number of nodes.

    Returns:
        Set of node indices that are loop headers.
    """
    if edge_index.numel() == 0 or num_nodes == 0:
        return set()

    parent: torch.Tensor = compute_dominator_tree(edge_index, num_nodes, entry_node=0)
    parent_list: List[int] = parent.tolist()

    def _dominates(dom_candidate: int, node: int) -> bool:
        """True if dom_candidate dominates node (ancestor in idom tree)."""
        visited_nodes: Set[int] = set()
        cur: int = node
        while cur != -1 and cur not in visited_nodes:
            if cur == dom_candidate:
                return True
            visited_nodes.add(cur)
            cur = parent_list[cur]
        return False

    headers: Set[int] = set()
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    for u, v in zip(src_list, dst_list):
        if _dominates(v, u):
            headers.add(v)

    return headers


# ═══════════════════════ structural feature extraction ═══════════════════════

def extract_structural_features(binary_cfg: BinaryCFG) -> torch.Tensor:
    """Compute the 12-dimensional structural feature vector for every basic block.

    The features follow the order defined in
    :py:meth:`BasicBlock.struct_features`:

    ========  ==========================================================
    Index     Feature
    ========  ==========================================================
    0         in-degree
    1         out-degree
    2         is_loop_header  (1.0 / 0.0)
    3         is_loop_exit    (1.0 / 0.0)
    4         dominance_depth
    5         post-dominance_depth
    6         normalised block size  (num_instrs / 64)
    7         instruction entropy
    8         has_call        (1.0 / 0.0)
    9         has_indirect_jump (1.0 / 0.0)
    10        DFS position    (normalised)
    11        BFS position    (normalised)
    ========  ==========================================================

    This function recomputes in-/out-degree, loop-header flags, and
    dominance depth from the CFG edge list so that precomputed fields are
    kept consistent even if they were never set.

    Args:
        binary_cfg: A ``BinaryCFG`` instance.

    Returns:
        features: Shape ``[num_blocks, 12]`` (float32).
    """
    num_blocks: int = binary_cfg.num_blocks

    if num_blocks == 0:
        return torch.zeros(0, 12, dtype=torch.float32)

    # Build edge_index and edge_type tensors
    if len(binary_cfg.edges) > 0:
        src_list: List[int] = [e[0] for e in binary_cfg.edges]
        dst_list: List[int] = [e[1] for e in binary_cfg.edges]
        etype_list: List[int] = [e[2] for e in binary_cfg.edges]
        edge_index: torch.Tensor = torch.tensor(
            [src_list, dst_list], dtype=torch.int64
        )
        edge_type_t: torch.Tensor = torch.tensor(etype_list, dtype=torch.int64)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.int64)
        edge_type_t = torch.zeros(0, dtype=torch.int64)

    # ---- in-degree / out-degree ----
    in_deg: torch.Tensor = torch.zeros(num_blocks, dtype=torch.float32)
    out_deg: torch.Tensor = torch.zeros(num_blocks, dtype=torch.float32)
    if edge_index.numel() > 0:
        for d in edge_index[1].tolist():
            in_deg[d] += 1.0
        for s in edge_index[0].tolist():
            out_deg[s] += 1.0

    # ---- loop headers ----
    loop_headers: Set[int] = find_loop_headers(edge_index, num_blocks)

    # ---- loop exits: blocks with an edge *out* of a loop (successor not dominated by any loop header) ----
    # Simple heuristic: a block is a loop exit if it is a successor of a
    # loop header's dominated region and has a successor outside that region.
    # For efficiency, mark blocks that have a back-edge source as their
    # predecessor (i.e. blocks right after the loop body).
    loop_exit_set: Set[int] = set()
    if edge_index.numel() > 0:
        parent_tensor: torch.Tensor = compute_dominator_tree(edge_index, num_blocks)
        parent_l: List[int] = parent_tensor.tolist()
        for s_node, d_node, et in binary_cfg.edges:
            # heuristic: a loop exit is a node whose out-degree > 1 and
            # one successor is a loop header (back-edge target)
            if d_node in loop_headers and et == int(EdgeType.BACK):
                # the *source* of the back edge is inside the loop; if it
                # has another successor outside, it is also a loop exit
                loop_exit_set.add(s_node)
        # also mark blocks whose block data says is_loop_exit
        for blk in binary_cfg.blocks:
            if blk.is_loop_exit:
                loop_exit_set.add(blk.block_id)

    # ---- dominance depth ----
    dom_depth: torch.Tensor = torch.zeros(num_blocks, dtype=torch.float32)
    if edge_index.numel() > 0:
        parent_tensor = compute_dominator_tree(edge_index, num_blocks, entry_node=0)
        parent_l = parent_tensor.tolist()
        for node in range(num_blocks):
            d: int = 0
            cur: int = node
            visited_set: Set[int] = set()
            while cur != parent_l[cur] and cur != -1 and cur not in visited_set:
                visited_set.add(cur)
                d += 1
                cur = parent_l[cur]
            dom_depth[node] = float(d)

    # ---- post-dominance depth (reverse CFG) ----
    post_dom_depth: torch.Tensor = torch.zeros(num_blocks, dtype=torch.float32)
    if edge_index.numel() > 0:
        rev_edge_index: torch.Tensor = torch.stack(
            [edge_index[1], edge_index[0]], dim=0
        )
        # Find exit nodes (nodes with out_degree == 0)
        exit_nodes: List[int] = [
            i for i in range(num_blocks) if out_deg[i].item() == 0.0
        ]
        exit_node: int = exit_nodes[0] if exit_nodes else num_blocks - 1
        pdom_parent: torch.Tensor = compute_dominator_tree(
            rev_edge_index, num_blocks, entry_node=exit_node
        )
        pdom_l: List[int] = pdom_parent.tolist()
        for node in range(num_blocks):
            d = 0
            cur = node
            visited_set = set()
            while cur != pdom_l[cur] and cur != -1 and cur not in visited_set:
                visited_set.add(cur)
                d += 1
                cur = pdom_l[cur]
            post_dom_depth[node] = float(d)

    # ---- DFS / BFS positions ----
    succs_map: Dict[int, List[int]] = {i: [] for i in range(num_blocks)}
    if edge_index.numel() > 0:
        for s_idx, d_idx in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            succs_map[s_idx].append(d_idx)

    # DFS order
    dfs_order: List[int] = []
    dfs_visited: Set[int] = set()

    def _iterative_dfs(start: int) -> None:
        stack: List[int] = [start]
        while stack:
            node = stack.pop()
            if node in dfs_visited:
                continue
            dfs_visited.add(node)
            dfs_order.append(node)
            for succ in reversed(succs_map[node]):
                if succ not in dfs_visited:
                    stack.append(succ)

    _iterative_dfs(0)
    # Handle disconnected nodes
    for i in range(num_blocks):
        if i not in dfs_visited:
            dfs_order.append(i)

    dfs_pos: torch.Tensor = torch.zeros(num_blocks, dtype=torch.float32)
    for rank, node in enumerate(dfs_order):
        dfs_pos[node] = float(rank) / max(num_blocks - 1, 1)

    # BFS order
    bfs_order: List[int] = []
    bfs_visited: Set[int] = set()
    bfs_queue: deque = deque([0])
    bfs_visited.add(0)
    while bfs_queue:
        node = bfs_queue.popleft()
        bfs_order.append(node)
        for succ in succs_map[node]:
            if succ not in bfs_visited:
                bfs_visited.add(succ)
                bfs_queue.append(succ)
    for i in range(num_blocks):
        if i not in bfs_visited:
            bfs_order.append(i)

    bfs_pos: torch.Tensor = torch.zeros(num_blocks, dtype=torch.float32)
    for rank, node in enumerate(bfs_order):
        bfs_pos[node] = float(rank) / max(num_blocks - 1, 1)

    # ---- assemble per-block feature vectors ----
    features: torch.Tensor = torch.zeros(num_blocks, 12, dtype=torch.float32)
    for i, blk in enumerate(binary_cfg.blocks):
        features[i, 0] = in_deg[i]
        features[i, 1] = out_deg[i]
        features[i, 2] = 1.0 if i in loop_headers else 0.0
        features[i, 3] = 1.0 if i in loop_exit_set else 0.0
        features[i, 4] = dom_depth[i]
        features[i, 5] = post_dom_depth[i]
        features[i, 6] = float(len(blk.instructions)) / 64.0
        features[i, 7] = blk.instruction_entropy
        features[i, 8] = 1.0 if blk.has_call else 0.0
        features[i, 9] = 1.0 if blk.has_indirect_jump else 0.0
        features[i, 10] = dfs_pos[i]
        features[i, 11] = bfs_pos[i]

    return features


# ═══════════════════════ region segmentation ═════════════════════════════════

def compute_region_segmentation(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[Dict]:
    """Segment the CFG into structural regions based on pattern labels.

    Groups nodes by their pattern label into contiguous (connected)
    regions.  Each region is characterised by the set of nodes it contains,
    its structural type, and (optionally) its header node.

    Algorithm:
        1. For each distinct pattern label, collect all nodes with that label.
        2. Within each label group, find connected components using the
           CFG adjacency (ignoring edge direction).
        3. For loop/if/switch regions, identify the header node (lowest
           DFS order among the group).

    Args:
        pattern_labels: Per-node pattern label.  Shape ``[num_nodes]`` (int64).
        edge_index:     Edge endpoints.  Shape ``[2, num_edges]`` (int64).
        num_nodes:      Number of nodes.

    Returns:
        List of dicts, each with keys:
            ``'nodes'``  -- ``List[int]``: node indices in this region.
            ``'type'``   -- ``str``: pattern type name
                           (e.g. ``'SEQUENTIAL'``, ``'IF_HEADER'``, ...).
            ``'header'`` -- ``Optional[int]``: header node index, or ``None``.
    """
    labels: List[int] = pattern_labels.tolist()

    # Build undirected graph for connectivity (NetworkX)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            G.add_edge(s, d)

    # Group nodes by pattern label
    label_groups: Dict[int, List[int]] = {}
    for node_idx, lab in enumerate(labels):
        label_groups.setdefault(lab, []).append(node_idx)

    # Pattern type names from the enum
    _pattern_names: Dict[int, str] = {pt.value: pt.name for pt in PatternType}

    # Header-type patterns
    _header_patterns: Set[int] = {
        PatternType.IF_HEADER,
        PatternType.LOOP_HEADER,
        PatternType.SWITCH_HEADER,
    }

    regions: List[Dict] = []

    for lab, node_list in label_groups.items():
        node_set: Set[int] = set(node_list)

        # Find connected components within this label group (NetworkX)
        subgraph = G.subgraph(node_list)
        for cc in nx.connected_components(subgraph):
            component = sorted(cc)  # deterministic order

            # Determine header
            header: Optional[int] = None
            if lab in _header_patterns:
                header = component[0]  # smallest index ~ earliest in DFS

            regions.append({
                "nodes": component,
                "type": lab,
                "header": header,
            })

    # Sort regions by the minimum node index for deterministic ordering
    regions.sort(key=lambda r: r["nodes"][0] if r["nodes"] else 0)
    return regions


def _natural_loop_nodes(G: nx.DiGraph, back_src: int, head: int) -> Set[int]:
    """Natural loop for back edge (back_src, head) with header *head*."""
    loop: Set[int] = {head}
    stack = [back_src]
    while stack:
        n = stack.pop()
        if n in loop:
            continue
        loop.add(n)
        for p in G.predecessors(n):
            if p not in loop:
                stack.append(p)
    return loop


def compute_binary_regions_dominator_tree(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[Dict]:
    """Doc §6.2.2 (binary): dominator + natural loops for loop bodies.

    1) For each back edge (u, v) where v dominates u, take the natural loop
       containing the header v.
    2) Remaining nodes are segmented with :func:`compute_region_segmentation`
       on the induced subgraph (same labels, connectivity).
    """
    if num_nodes == 0:
        return []
    G = _edge_index_to_nx_digraph(edge_index, num_nodes)
    assigned: Set[int] = set()
    regions: List[Dict] = []

    if edge_index.numel() > 0:
        idom = compute_dominator_tree(edge_index, num_nodes, 0)
        idl = idom.tolist()

        def dominates(u: int, v: int) -> bool:
            cur = v
            visited: Set[int] = set()
            while cur != -1 and cur < len(idl) and cur not in visited:
                if cur == u:
                    return True
                visited.add(cur)
                cur = idl[cur]
            return False

        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if dominates(v, u) and v != u:
                loop = _natural_loop_nodes(G, u, v)
                nodes = sorted(loop)
                if nodes:
                    regions.append({
                        "nodes": nodes,
                        "type": int(PatternType.LOOP_HEADER),
                        "header": v,
                    })
                    assigned.update(nodes)

    rem = [i for i in range(num_nodes) if i not in assigned]
    if rem:
        inv = sorted(rem)
        remap = {old: k for k, old in enumerate(inv)}
        rem_set = set(rem)
        es: List[List[int]] = []
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                if s in rem_set and d in rem_set:
                    es.append([remap[s], remap[d]])
        if es:
            ei = torch.tensor(es, dtype=torch.long).t()
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        sub_labels = pattern_labels[rem]
        sub_regions = compute_region_segmentation(sub_labels, ei, len(rem))
        for reg in sub_regions:
            old_nodes = [inv[n] for n in reg["nodes"]]
            regions.append({
                "nodes": old_nodes,
                "type": reg["type"],
                "header": reg.get("header"),
            })

    regions.sort(key=lambda r: r["nodes"][0] if r["nodes"] else 0)
    return regions


def compute_source_regions_ast(
    ast_types: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[Dict]:
    """Doc §6.2.2 (source): AST compound / structural statement types.

    Groups nodes by AST label into connected components; headers are
    IF / loop / switch / compound statement types.
    """
    if num_nodes == 0:
        return []
    labels = ast_types.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            G.add_edge(s, d)

    label_groups: Dict[int, List[int]] = {}
    for node_idx, lab in enumerate(labels):
        label_groups.setdefault(lab, []).append(node_idx)

    _header_ast: Set[int] = {
        int(ASTType.IF_STMT),
        int(ASTType.WHILE_STMT),
        int(ASTType.FOR_STMT),
        int(ASTType.DO_WHILE_STMT),
        int(ASTType.SWITCH_STMT),
        int(ASTType.COMPOUND_STMT),
    }

    regions: List[Dict] = []

    for lab, node_list in label_groups.items():
        subgraph = G.subgraph(node_list)
        for cc in nx.connected_components(subgraph):
            component = sorted(cc)
            header: Optional[int] = None
            if lab in _header_ast:
                header = component[0]
            regions.append({
                "nodes": component,
                "type": lab,
                "header": header,
            })

    regions.sort(key=lambda r: r["nodes"][0] if r["nodes"] else 0)
    return regions


def compute_source_regions_compound(
    ast_types: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    compound_parent_ids: torch.Tensor,
) -> List[Dict]:
    """Doc §6.2.2 (source): 按构建时写入的 compound 父语句列下标分组，再在组内按 CFG 边拆连通分量。

    ``compound_parent_ids[i] == -1`` 的节点回退为 :func:`compute_source_regions_ast`。
    """
    if num_nodes == 0:
        return []
    cids = compound_parent_ids.tolist()
    labels = ast_types.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            G.add_edge(s, d)

    regions: List[Dict] = []
    assigned: Set[int] = set()

    pos_by_cid: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(cids):
        if c >= 0:
            pos_by_cid[c].append(i)

    for _cid, nodes in pos_by_cid.items():
        if not nodes:
            continue
        sub = G.subgraph(nodes)
        for cc in nx.connected_components(sub):
            component = sorted(cc)
            h = component[0]
            regions.append({
                "nodes": component,
                "type": labels[h],
                "header": h,
            })
            assigned.update(component)

    rem = [i for i in range(num_nodes) if cids[i] < 0]
    if rem:
        rem_set = set(rem)
        es: List[List[int]] = []
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                if s in rem_set and d in rem_set:
                    es.append([s, d])
        inv = sorted(rem)
        remap = {old: k for k, old in enumerate(inv)}
        if es:
            ei = torch.tensor(
                [[remap[e[0]], remap[e[1]]] for e in es],
                dtype=torch.long,
            ).t()
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        sub_ast = ast_types[torch.tensor(rem, dtype=torch.long, device=ast_types.device)]
        sub_regions = compute_source_regions_ast(sub_ast, ei, len(rem))
        for reg in sub_regions:
            old_nodes = [inv[n] for n in reg["nodes"]]
            regions.append({
                "nodes": old_nodes,
                "type": reg["type"],
                "header": reg.get("header"),
            })

    regions.sort(key=lambda r: r["nodes"][0] if r["nodes"] else 0)
    return regions


def compute_immediate_post_dominator(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Immediate post-dominator（反向 CFG 上从出口求 idom）。"""
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.int64)
    out_deg = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.numel() > 0:
        for s in edge_index[0].tolist():
            out_deg[s] += 1
    exits = [i for i in range(num_nodes) if out_deg[i].item() == 0]
    exit_node = exits[0] if exits else num_nodes - 1
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return compute_dominator_tree(rev, num_nodes, entry_node=exit_node)


def _collect_cfg_nodes_until_merge(
    G: nx.DiGraph,
    roots: List[int],
    merge: int,
) -> Set[int]:
    """从 *roots* 沿前向边可达、在到达 *merge* 前停止的节点集。"""
    reg: Set[int] = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n == merge or n in reg:
            continue
        reg.add(n)
        for t in G.successors(n):
            if t != merge:
                stack.append(t)
    return reg


def compute_binary_regions_structural(
    pattern_labels: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[Dict]:
    """§6.2.2 二进制侧：loop / if / switch 头结点区域 + 未分配 sequential。"""
    if num_nodes == 0:
        return []
    labels = pattern_labels.long().tolist()
    G = _edge_index_to_nx_digraph(edge_index, num_nodes)
    idom = compute_dominator_tree(edge_index, num_nodes, 0)
    idom_l = idom.tolist()
    ipost_l = compute_immediate_post_dominator(edge_index, num_nodes).tolist()

    def dominates(a: int, b: int) -> bool:
        cur = b
        seen: Set[int] = set()
        while cur != -1 and cur < len(idom_l) and cur not in seen:
            if cur == a:
                return True
            seen.add(cur)
            cur = idom_l[cur]
        return False

    assigned: Set[int] = set()
    regions: List[Dict] = []

    if edge_index.numel() > 0:
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if v < 0 or u < 0:
                continue
            if dominates(v, u) and v != u and labels[v] == int(PatternType.LOOP_HEADER):
                loop = _natural_loop_nodes(G, u, v)
                nodes = sorted(loop)
                if nodes:
                    regions.append({
                        "nodes": nodes,
                        "type": int(PatternType.LOOP_HEADER),
                        "header": v,
                    })
                    assigned.update(nodes)

    for h in range(num_nodes):
        if labels[h] != int(PatternType.IF_HEADER) or h in assigned:
            continue
        merge = ipost_l[h]
        if merge < 0 or merge == h:
            continue
        roots = [s for s in G.successors(h)]
        body = _collect_cfg_nodes_until_merge(G, roots, merge)
        region_nodes = sorted({h} | body)
        if region_nodes:
            regions.append({
                "nodes": region_nodes,
                "type": int(PatternType.IF_HEADER),
                "header": h,
            })
            assigned.update(region_nodes)

    for h in range(num_nodes):
        if labels[h] != int(PatternType.SWITCH_HEADER) or h in assigned:
            continue
        merge = ipost_l[h]
        roots = [s for s in G.successors(h)]
        if not roots:
            continue
        if merge < 0 or merge == h:
            body: Set[int] = set()
            stack = list(roots)
            while stack:
                n = stack.pop()
                if n in body:
                    continue
                body.add(n)
                for t in G.successors(n):
                    stack.append(t)
        else:
            body = _collect_cfg_nodes_until_merge(G, roots, merge)
        region_nodes = sorted({h} | body)
        if region_nodes:
            regions.append({
                "nodes": region_nodes,
                "type": int(PatternType.SWITCH_HEADER),
                "header": h,
            })
            assigned.update(region_nodes)

    rem = [i for i in range(num_nodes) if i not in assigned]
    if rem:
        inv = sorted(rem)
        remap = {old: k for k, old in enumerate(inv)}
        rem_set = set(rem)
        es: List[List[int]] = []
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                if s in rem_set and d in rem_set:
                    es.append([remap[s], remap[d]])
        if es:
            ei = torch.tensor(es, dtype=torch.long).t()
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        sub_labels = pattern_labels[rem]
        sub_regions = compute_region_segmentation(sub_labels, ei, len(rem))
        for reg in sub_regions:
            old_nodes = [inv[n] for n in reg["nodes"]]
            regions.append({
                "nodes": old_nodes,
                "type": reg["type"],
                "header": reg.get("header"),
            })

    regions.sort(key=lambda r: r["nodes"][0] if r["nodes"] else 0)
    return regions


