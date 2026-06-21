"""
StrucLift: Multi-Granularity Structural Alignment and Structure-Conditioned
LLM Decoding for Neural Decompilation.

Modules
-------
  Module A  — Structure-Aware CFG Encoders (binary + source)
  Module B  — Multi-Granularity Cross-Graph Alignment (SCOT)
  Module C  — Structure-Conditioned LLM Decoder
  Module D  — Structural Consistency RL (GRPO)

Usage
-----
    from struclift.config import StrucLiftConfig
    from struclift.models.struclift import StrucLift

    config = StrucLiftConfig()
    model = StrucLift(config)
"""

__version__ = "0.1.0"
