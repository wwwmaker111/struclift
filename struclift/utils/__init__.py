"""
StrucLift utility modules.

* :mod:`~struclift.utils.sinkhorn`    -- Sinkhorn OT solvers & SCOT
* :mod:`~struclift.utils.graph_utils` -- CFG graph analysis utilities
* :mod:`~struclift.utils.skeleton`    -- CST construction & skeleton generation
"""

from .sinkhorn import (
    log_sinkhorn,
    sinkhorn_from_cost,
    structure_constrained_ot,
    hungarian_matching,
)

from .graph_utils import (
    build_adjacency_matrix,
    compute_dominator_tree,
    find_loop_headers,
    extract_structural_features,
    compute_region_segmentation,
)

from .skeleton import (
    CSTNode,
    build_bb_opcode_hints_from_batch,
    build_control_structure_tree,
    build_control_structure_tree_with_stats,
    build_skeleton_for_binary_cfg,
    compute_pattern_reliability,
    enrich_skeleton_slots_with_alignment_matrix,
    serialize_cst_to_skeleton,
    build_slot_mask,
)
