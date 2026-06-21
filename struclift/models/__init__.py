"""
StrucLift model modules.

Module A: Structure-Aware Binary CFG Encoder & Source CFG Encoder.
Module B: Multi-Granularity Cross-Graph Structural Alignment.
Module C: Structure-Conditioned LLM Decoder.
Module D: Structural Consistency Reinforcement Learning.

Note: Module A and Module B require ``torch_geometric``.  If it is not
installed, importing from those submodules will raise ``ImportError``, but
Modules C, D, and the rest of the package remain usable.
"""

import importlib
from typing import Any

# Lazy-import registry: module_name → list of public names
_LAZY_IMPORTS = {
    ".module_a": [
        "InstructionEmbedding", "BlockTransformer", "AttentivePooling",
        "StructuralFeatureMLP", "EdgeTypedGATLayer", "SubgraphPatternClassifier",
        "PMAPooling", "BinaryCFGEncoder", "SourceCFGEncoder",
    ],
    ".module_b": [
        "RegionAligner", "CrossAttentionLayer", "CrossAttentionRefinement",
        "CrossGraphAlignmentModule", "ModuleBOutput",
    ],
    ".module_c": [
        "CrossAttentionAdapter", "StructureConditionedDecoder",
        "SlotWeightBuilder", "AdapterInjector",
    ],
    ".module_d": [
        "compile_reward", "structural_reward", "semantic_reward",
        "combined_reward", "GRPOConfig", "GRPOTrainer",
    ],
}

# build reverse mapping: name → relative module
_NAME_TO_MODULE = {}
for _mod, _names in _LAZY_IMPORTS.items():
    for _name in _names:
        _NAME_TO_MODULE[_name] = _mod

__all__ = list(_NAME_TO_MODULE.keys())


def __getattr__(name: str) -> Any:
    if name in _NAME_TO_MODULE:
        mod = importlib.import_module(_NAME_TO_MODULE[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
