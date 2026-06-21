#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic contract smoke for Module-B v3.

This is intentionally small and deterministic.  It checks the first-version
contract from ``module_b_v3_practical_candidate_ranker_strategy.md`` before
running expensive real-data audits:

* candidate generation always includes a fallback candidate;
* selected candidates preserve BB coverage and CFG edges;
* explicit fallback/goto/escape slots keep invalid structure from leaking out;
* the ranker cannot select a candidate outside the validator safety envelope;
* validator detects malformed candidates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, rel_path: str) -> Any:
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MVP = load_module("module_b_v3_mvp_contract_smoke", "struclift/utils/module_b_v3_mvp.py")
RANKER = load_module("module_b_v3_ranker_contract_smoke", "struclift/utils/module_b_v3_ranker.py")


class ZeroRanker:
    weights: List[float] = []

    def score_candidate(self, _cand: Dict[str, Any], _cfg_feat: Dict[str, Any]) -> float:
        return 0.0


def make_candidate_record(cfg: Any, cand: Any, cid: int) -> Dict[str, Any]:
    rendered = MVP.render_candidate(cand)
    report = MVP.validate_candidate(cfg, cand, rendered_text=rendered).asdict()
    slots = [s.asdict() for s in cand.slots]
    rec: Dict[str, Any] = {
        "candidate_id": int(cid),
        "name": cand.name,
        "slots": slots,
        "outline": [[k, d] for k, d in MVP.candidate_outline(cand)],
        "validator_report": report,
        "edge_audit": {"strict_edge_preservation_ok": bool(report.get("strict_edge_preservation_ok"))},
        "grounding": {"slot_grounding_pointer_acc": 1.0},
        "rank_target": {
            "hard_valid": bool(report.get("valid") and report.get("strict_edge_preservation_ok")),
        },
    }
    rec["features"] = RANKER.candidate_features(rec, RANKER.cfg_features(cfg))
    return rec


def assert_selected_contract(name: str, cfg: Any, cands: List[Any], failures: List[Tuple[Any, ...]]) -> None:
    if not any(c.name == "full_fallback" for c in cands):
        failures.append((name, "missing_full_fallback"))
    selected, report, _text = MVP.select_candidate(cfg, cands)
    if not (
        report.valid
        and report.bb_coverage_ok
        and report.edge_preservation_ok
        and report.parse_success
        and not report.fatal_errors
    ):
        failures.append((name, "select_invalid", selected.name, report.asdict()))

    cfg_feat = RANKER.cfg_features(cfg)
    recs = [make_candidate_record(cfg, cand, cid) for cid, cand in enumerate(cands)]
    ranked = RANKER.select_with_ranker(recs, cfg_feat, ZeroRanker())
    if ranked is None:
        failures.append((name, "ranker_selected_none"))
    else:
        rep = ranked.get("validator_report") or {}
        if not (
            rep.get("valid")
            and rep.get("bb_coverage_ok")
            and rep.get("edge_preservation_ok")
            and rep.get("parse_success")
        ):
            failures.append((name, "ranker_selected_invalid", ranked.get("name"), rep))

    print(
        f"{name}: n_bb={cfg.n_bb} candidates={len(cands)} selected={selected.name} "
        f"ranked={(ranked or {}).get('name')} goto={report.goto_count} "
        f"break={report.break_count} continue={report.continue_count} "
        f"fallback_bbs={report.fallback_bb_count} sese={len(cfg.sese_regions)} "
        f"irreducible={len(cfg.irreducible_sccs)}"
    )


def assert_negative_validator(failures: List[Tuple[Any, ...]]) -> None:
    cfg = MVP.analyse_cfg({"bin_block_lengths": [1, 1, 1], "bin_edge_index": [[0, 1], [1, 2]]})

    b = MVP._Builder(cfg)
    bad_root = MVP.Node(kind="FUNCTION", children=[b.block([0], role="bb")], region_bbs=[0])
    bad = MVP.Candidate("bad_missing", bad_root, b.slots, set(), [], [])
    rep = MVP.validate_candidate(cfg, bad, rendered_text=MVP.render_candidate(bad)).asdict()
    if (
        rep.get("valid")
        or "missing_bb_coverage" not in (rep.get("fatal_errors") or [])
        or "uncovered_cfg_edge" not in (rep.get("fatal_errors") or [])
    ):
        failures.append(("negative_missing_or_edge_not_detected", rep))

    b2 = MVP._Builder(cfg)
    dup_root = MVP.Node(
        kind="FUNCTION",
        children=[
            b2.block([0], role="bb"),
            b2.block([0], role="bb"),
            b2.block([1], role="bb"),
            b2.block([2], role="bb"),
        ],
        region_bbs=[0, 1, 2],
    )
    dup = MVP.Candidate("bad_dup", dup_root, b2.slots, set(cfg.edges), [], [])
    rep2 = MVP.validate_candidate(cfg, dup, rendered_text=MVP.render_candidate(dup)).asdict()
    if rep2.get("valid") or "duplicate_primary_bb" not in (rep2.get("fatal_errors") or []):
        failures.append(("negative_duplicate_not_detected", rep2))


def main() -> None:
    cases: Dict[str, Dict[str, Any]] = {
        "empty": {"bin_block_lengths": [], "bin_edge_index": [[], []]},
        "single": {"bin_block_lengths": [1], "bin_edge_index": [[], []]},
        "linear4": {"bin_block_lengths": [1, 1, 1, 1], "bin_edge_index": [[0, 1, 2], [1, 2, 3]]},
        "if_else": {
            "bin_block_lengths": [1, 1, 1, 1],
            "bin_edge_index": [[0, 0, 1, 2], [1, 2, 3, 3]],
        },
        "while_exit": {"bin_block_lengths": [1, 1, 1], "bin_edge_index": [[0, 0, 1], [1, 2, 0]]},
        "loop_continue_break": {
            "bin_block_lengths": [1, 1, 1, 1, 1],
            "bin_edge_index": [[0, 1, 1, 2, 2, 3], [1, 2, 4, 1, 3, 1]],
        },
        "cross_edge": {
            "bin_block_lengths": [1, 1, 1, 1, 1],
            "bin_edge_index": [[0, 0, 1, 2, 3, 1], [1, 2, 3, 3, 4, 4]],
        },
        "irreducible_like": {
            "bin_block_lengths": [1, 1, 1, 1],
            "bin_edge_index": [[0, 0, 1, 2, 2], [1, 2, 2, 1, 3]],
        },
    }

    failures: List[Tuple[Any, ...]] = []
    for name, row in cases.items():
        cfg = MVP.analyse_cfg(row)
        cands = MVP.generate_candidates(cfg, candidate_beam=64, include_switch_chain=True)
        assert_selected_contract(name, cfg, cands, failures)
    assert_negative_validator(failures)

    if failures:
        print("FAILURES")
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("ALL_CONTRACT_AND_NEGATIVE_SMOKE_PASS")


if __name__ == "__main__":
    main()
