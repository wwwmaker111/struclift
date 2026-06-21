"""Load checkpoints when only part of tensors match current ``nn.Module`` shapes."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

__all__ = ["load_state_dict_skip_shape_mismatch"]


def load_state_dict_skip_shape_mismatch(
    model: nn.Module,
    state: Dict[str, Any],
    *,
    log: logging.Logger | None = None,
) -> tuple[list[str], list[str]]:
    """Load keys present in ``model.state_dict()`` whose tensors match shape (and coerce dtype).

    Skips mismatched shapes (logged). Use when upgrading Stage2 ckpt trained with another
    ``Module C`` / ``d_llm`` onto a larger LLM: encoders + alignment typically still load.

    Parameters
    ----------
    state
        Flat ``state_dict`` (not the Stage2 trainer wrapper dict).
    """
    lg = log or logging.getLogger(__name__)
    model_sd = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for k, ckpt_tensor in state.items():
        if k not in model_sd:
            continue
        if not isinstance(ckpt_tensor, torch.Tensor):
            continue
        target = model_sd[k]
        if ckpt_tensor.shape != target.shape:
            skipped.append(
                f"{k}: ckpt{tuple(ckpt_tensor.shape)} vs model{tuple(target.shape)}",
            )
            continue
        if ckpt_tensor.dtype != target.dtype:
            ckpt_tensor = ckpt_tensor.to(dtype=target.dtype)
        filtered[k] = ckpt_tensor

    if skipped:
        n = len(skipped)
        head = skipped[:12]
        lg.warning(
            "checkpoint 与当前模型形状不一致已跳过 %d 项"
            "（常见：decoder / adapter 随 --llm hidden_size 与 Stage2 时 d_llm 不同）。示例:%s%s",
            n,
            "".join("\n    " + s for s in head),
            "\n    ..." if n > len(head) else "",
        )

    inc = model.load_state_dict(filtered, strict=False)
    mk = getattr(inc, "missing_keys", None)
    uk = getattr(inc, "unexpected_keys", None)
    if mk is not None and uk is not None:
        return list(mk), list(uk)
    return list(inc[0]), list(inc[1])  # pragma: no cover
