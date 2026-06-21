#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3（binskel jsonl）训练入口：**不修改 struclift 核心代码**，仅在脚本内根据 HF
``AutoConfig`` 推断 ``Module C`` 的 ``d_llm`` / ``num_llm_layers``，便于频繁更换基座。

适用场景
--------
- 多模态/复合 config（如 ``text_config.hidden_size``）与纯文本 ``hidden_size`` 均可。
- 加载顺序：``AutoModelForCausalLM`` → ``AutoModelForConditionalGeneration`` → ``AutoModel``。

要求
----
与 ``train_stage3_binskel.py`` 相同：jsonl 须含 Stage3 字段；**tokenizer 与 sft_input_ids 必须同源**。

示例::

  CUDA_VISIBLE_DEVICES=0 python scripts/train_stage3_binskel_autoconfig.py \\
    --llm /data/chaoni/WQM/models/qwen3.6_27B \\
    --jsonl data/sft.jsonl \\
    --init-from checkpoints/stage2/best.pt \\
    --save-dir checkpoints/stage3_qwen \\
    --dry-run

可选：手动覆盖维度（一般不必）::

  python scripts/train_stage3_binskel_autoconfig.py ... --d-llm 5120 --num-llm-layers 64

多卡 LLM + 一条真实样本 dry-run（需 pip install accelerate；``CUDA_VISIBLE_DEVICES`` 指明物理卡）::

  CUDA_VISIBLE_DEVICES=3,4 python scripts/train_stage3_binskel_autoconfig.py \\
    --dry-run --llm-device-map-auto --llm-per-gpu-max-gib 38 \\
    --jsonl ... --init-from ... --llm ...

  同上设备 **正式 Stage3Trainer 训练**（27B + 每张卡显存配额紧、且保持 ``max_sft_seq_len`` 不降时建议使用）::

  CUDA_VISIBLE_DEVICES=3,4 python scripts/train_stage3_binskel_autoconfig.py \\
    --llm-device-map-auto --llm-per-gpu-max-gib 38 \\
    --jsonl ... --init-from ... --llm ... --save-dir checkpoints/stage3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    describe_binskel_batch,
    make_binskel_stage3_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.module_c import SlotWeightBuilder
from struclift.models.struclift import StrucLift
from struclift.training.stage3_sft import Stage3Trainer
from struclift.utils.state_dict_compat import load_state_dict_skip_shape_mismatch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_module_c_dims(hf_cfg) -> tuple[int, int]:
    """从 AutoConfig 推断 (d_llm, num_llm_layers)。"""
    tc = getattr(hf_cfg, "text_config", None)
    if tc is not None:
        hs = getattr(tc, "hidden_size", None)
        nl = getattr(tc, "num_hidden_layers", None)
        if hs is not None and nl is not None:
            return int(hs), int(nl)
    hs = getattr(hf_cfg, "hidden_size", None)
    nl = getattr(hf_cfg, "num_hidden_layers", None)
    if hs is not None and nl is not None:
        return int(hs), int(nl)
    raise ValueError(
        "无法从 AutoConfig 推断 d_llm / num_llm_layers。"
        "请检查 config（text_config 或顶层 hidden_size / num_hidden_layers），"
        "或使用命令行 --d-llm / --num-llm-layers。"
    )


def _apply_config_dims(cfg: StrucLiftConfig, hf_cfg, d_llm: int | None, n_layers: int | None) -> None:
    d_inf, n_inf = _infer_module_c_dims(hf_cfg)
    d = int(d_llm) if d_llm is not None else d_inf
    n = int(n_layers) if n_layers is not None else n_inf
    cfg.module_c.d_llm = d
    cfg.module_c.num_llm_layers = n
    if d_llm is not None or n_layers is not None:
        logger.info(
            "Module C: d_llm=%d num_llm_layers=%d（CLI 部分/全部覆盖；推断值为 %d,%d）",
            d,
            n,
            d_inf,
            n_inf,
        )
    else:
        logger.info(
            "从 AutoConfig 推断 Module C: d_llm=%d num_llm_layers=%d (model_type=%s)",
            d,
            n,
            getattr(hf_cfg, "model_type", "?"),
        )


def _build_llm_max_memory(per_gpu_max_gib: float | None) -> dict[int, str] | None:
    if per_gpu_max_gib is None or per_gpu_max_gib <= 0:
        return None
    import torch

    out: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        out[i] = f"{float(per_gpu_max_gib):g}GiB"
    return out


def _dispatch_safe_expand_token_vocab(llm: nn.Module, new_num_tokens: int) -> bool:
    """在 ``device_map=auto`` 分片图上替代 ``resize_token_embeddings``（易导致 CUDA illegal access）。

    在 ``embedding.weight`` / ``lm_head.weight`` **各自所处的设备上**就地行方向拼接，
    新权重用小方差正态初始化。
    Returns True 表示已完成扩充。"""
    inp = llm.get_input_embeddings()
    w = inp.weight
    if not isinstance(w, torch.Tensor):
        return False
    v_old = int(w.shape[0])
    h = int(w.shape[1])
    if new_num_tokens <= v_old:
        return False
    delta = new_num_tokens - v_old
    dev = w.device
    dt = w.dtype
    add = torch.empty(delta, h, dtype=dt, device=dev)
    add.normal_(mean=0.0, std=0.02)
    with torch.no_grad():
        big = torch.cat([w.detach(), add], dim=0).contiguous()
        inp.weight = nn.Parameter(big, requires_grad=w.requires_grad)

    tie = bool(
        getattr(getattr(llm, "config", None), "tie_word_embeddings", False),
    )

    lm = getattr(llm, "lm_head", None)
    if (not tie) and lm is not None and getattr(lm, "weight", None) is not None:
        lw = lm.weight
        if lw.dim() != 2:
            raise RuntimeError(
                "lm_head.weight 维度非 2：请离线 merge tokenizer 后再加载。",
            )
        # 常见：`Linear(hidden→vocab)` → weight [vocab, hidden]
        if int(lw.shape[0]) == v_old:
            ldev = lw.device
            ldt = lw.dtype
            ladd = torch.empty(delta, int(lw.shape[1]), dtype=ldt, device=ldev)
            ladd.normal_(mean=0.0, std=0.02)
            with torch.no_grad():
                lbig = torch.cat([lw.detach(), ladd], dim=0).contiguous()
                lm.weight = nn.Parameter(lbig, requires_grad=lw.requires_grad)
        else:
            raise RuntimeError(
                "lm_head.weight 形状与嵌入表不匹配 "
                f"(lm_head={tuple(lw.shape)}, vocab_old={v_old})；请离线 merge。",
            )

    if getattr(llm, "config", None) is not None:
        cfg = llm.config
        if getattr(cfg, "vocab_size", None) is not None:
            cfg.vocab_size = new_num_tokens
        tc = getattr(cfg, "text_config", None)
        if tc is not None and getattr(tc, "vocab_size", None) is not None:
            tc.vocab_size = new_num_tokens

    logger.info(
        "dispatch-safe 扩充词表: %d -> %d（跳 HF resize_token_embeddings）",
        v_old,
        new_num_tokens,
    )
    return True


def _load_hf_llm_autoconfig(
    cfg: StrucLiftConfig,
    llm_name_or_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
    *,
    device_map_auto: bool = False,
    llm_max_memory_per_gpu_gib: float | None = None,
    llm_dtype: str = "auto",
) -> nn.Module:
    """加载 HF 底座；可选 ``device_map=\"auto\"`` 多卡切分 LLM。

    ``device_map_auto=True`` 时不在末尾 ``.to(device)``，由 accelerate 占位各层。
    """
    path = llm_name_or_path
    if llm_dtype == "auto":
        if device.type != "cuda":
            dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif llm_dtype == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            logger.warning(
                "--llm-dtype bf16 requested, but torch.cuda.is_bf16_supported() is false; "
                "falling back to fp16 to avoid CUDA driver invalid-argument errors.",
            )
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    elif llm_dtype == "fp16":
        dtype = torch.float16
    elif llm_dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported --llm-dtype: {llm_dtype}")
    logger.info("加载 LLM dtype=%s", dtype)

    from transformers import AutoModel, AutoModelForCausalLM

    llm: nn.Module | None = None
    load_errors: list[str] = []

    if device_map_auto:
        try:
            import accelerate  # noqa: F401
        except ImportError:
            logger.error(
                "--llm-device-map-auto 需要 accelerate（pip install accelerate）",
            )
            raise SystemExit(1)
        if device.type != "cuda":
            logger.error("--llm-device-map-auto 仅在 CUDA 下可用")
            raise SystemExit(1)
        mm = _build_llm_max_memory(llm_max_memory_per_gpu_gib)
        kw = dict(
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if mm is not None:
            kw["max_memory"] = mm
        try:
            llm = AutoModelForCausalLM.from_pretrained(path, **kw)
            logger.info(
                "已用 AutoModelForCausalLM + device_map=auto 加载基座: %s",
                path,
            )
            dm = getattr(llm, "hf_device_map", None)
            if isinstance(dm, dict):
                logger.info(
                    "hf_device_map 设备集合=%s（%d 个组件）",
                    sorted({str(v) for v in dm.values()}),
                    len(dm),
                )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"device_map=auto 加载失败: {e!r}") from e
    else:
        try:
            from transformers import AutoModelForConditionalGeneration as AutoModelForCondGen
        except ImportError:
            AutoModelForCondGen = None  # type: ignore[misc,assignment]

        loaders: list[tuple[str, object]] = [
            (
                "AutoModelForCausalLM",
                lambda: AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=True, torch_dtype=dtype,
                ),
            ),
        ]
        if AutoModelForCondGen is not None:
            loaders.append(
                (
                    "AutoModelForConditionalGeneration",
                    lambda: AutoModelForCondGen.from_pretrained(
                        path, trust_remote_code=True, torch_dtype=dtype,
                    ),
                ),
            )
        loaders.append(
            (
                "AutoModel",
                lambda: AutoModel.from_pretrained(
                    path, trust_remote_code=True, torch_dtype=dtype,
                ),
            ),
        )
        for name, factory in loaders:
            try:
                llm = factory()
                logger.info("已用 %s 加载基座: %s", name, path)
                break
            except Exception as e:  # noqa: BLE001 —— 依次回退
                load_errors.append(f"{name}: {e!r}")

    if llm is None:
        msg = "无法用 AutoModel* 加载基座。尝试过:\n  " + "\n  ".join(load_errors)
        raise RuntimeError(msg)

    if target_vocab_size is not None and target_vocab_size > 0:
        emb = llm.get_input_embeddings()
        cur = emb.num_embeddings
        if target_vocab_size != cur:
            if device_map_auto:
                # HF resize_token_embeddings 在 accelerate/device_map 上易触发 CUDA illegal access
                ok = _dispatch_safe_expand_token_vocab(llm, int(target_vocab_size))
                if not ok:
                    raise RuntimeError(
                        "device_map_auto 模式下无法扩充词表：请离线 merge tokenizer 与权重。",
                    )
            else:
                try:
                    llm.resize_token_embeddings(
                        target_vocab_size, mean_resizing=False,
                    )
                except TypeError:
                    llm.resize_token_embeddings(target_vocab_size)
                logger.info(
                    "resize_token_embeddings: %d -> %d（与 tokenizer 长度对齐）",
                    cur,
                    target_vocab_size,
                )

    try:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=cfg.module_c.lora_r,
            lora_alpha=cfg.module_c.lora_alpha,
            lora_dropout=cfg.module_c.lora_dropout,
            target_modules=list(cfg.module_c.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lc)
        logger.info("已启用 PEFT LoRA (r=%d)", cfg.module_c.lora_r)
    except ImportError:
        logger.warning("未安装 peft：冻结 LLM 基座，仅训练 CrossAttentionAdapter")
        for p in llm.parameters():
            p.requires_grad = False

    if not device_map_auto:
        try:
            llm = llm.to(device)
        except RuntimeError as e:
            raise RuntimeError(
                f"LLM move to {device} failed after LoRA injection. "
                "If this is a CUDA driver invalid-argument error, retry with "
                "--llm-dtype fp16 or --llm-dtype auto, and check that CUDA_VISIBLE_DEVICES "
                "maps the requested --device correctly."
            ) from e

    # 冒烟：forward 必须带 .logits
    probe_dev = llm.get_input_embeddings().weight.device
    llm.eval()
    with torch.no_grad():
        probe = torch.ones(1, 2, dtype=torch.long, device=probe_dev)
        attn = torch.ones_like(probe)
        try:
            try:
                out = llm(input_ids=probe, attention_mask=attn)
            except TypeError:
                out = llm(input_ids=probe)
        except RuntimeError as e:
            raise RuntimeError(
                f"LLM CUDA probe failed on device={probe_dev}, dtype={dtype}. "
                "Retry with --llm-dtype fp16/auto, or use --llm-device-map-auto "
                "if the single visible GPU is tight or unstable."
            ) from e
    if not hasattr(out, "logits"):
        raise RuntimeError(
            "基座前向返回无 .logits，与 StructureConditionedDecoder 不兼容。"
        )
    llm.train()
    return llm


def _align_decoder_adapters_with_llm_shards(dec: nn.Module) -> None:
    """device_map LLM：每层 adapter / graph_proj 与对应层嵌入设备对齐。"""
    from struclift.models.module_c import AdapterInjector

    layers = AdapterInjector._find_layers(dec._llm)  # type: ignore[attr-defined]
    for i, layer in enumerate(layers):
        ld = next(layer.parameters()).device
        dec.adapters[i] = dec.adapters[i].to(ld)  # type: ignore[attr-defined]
    emb_dev = dec._llm.get_input_embeddings().weight.device  # type: ignore[union-attr]
    dec.graph_proj.to(emb_dev)  # type: ignore[attr-defined]
    logger.info(
        "decoder: graph_proj 与 embedding 同设备 %s；各 CrossAttentionAdapter 已对齐所属 layer",
        emb_dev,
    )


def _move_struclift_encoders_alignment_to_cuda0(model: StrucLift) -> None:
    d0 = torch.device("cuda:0")
    model.binary_encoder.to(d0)
    model.source_encoder.to(d0)
    model.alignment_module.to(d0)


def _maybe_enable_llm_gradient_checkpoint(model: nn.Module, cfg: StrucLiftConfig) -> None:
    """与 struclift.main.run_stage3 一致：注入后开启 HF LLM 的 activation checkpoint。"""
    if not getattr(cfg.training, "llm_gradient_checkpointing", False):
        return
    decoder = getattr(model, "decoder", None)
    inner = getattr(decoder, "_llm", None) if decoder is not None else None
    if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable()
        logger.info("已启用 LLM gradient checkpointing（降低激活显存）")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage3 SFT：根据 HF AutoConfig 自动设置 Module C 维度（不换核心库）",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        help="含 sft_input_ids / sft_labels / slot_types 的 binskel .jsonl",
    )
    src.add_argument(
        "--jsonl-list",
        type=Path,
        metavar="PATH.txt",
        help="每行一个 jsonl 路径（# 开头为注释）",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Stage2（或含 A+B）checkpoint .pt；与 --resume 二选一（resume 优先）",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "从 Stage3 完整 ckpt 续训（latest_stage3.pt / best_stage3.pt）："
            "模型 + 优化器 + epoch + best_loss；与 --init-from 互斥时优先 resume"
        ),
    )
    ap.add_argument(
        "--llm",
        type=str,
        required=True,
        help="HF 模型名或本地目录（必填；用于 AutoConfig + 加载权重）",
    )
    ap.add_argument(
        "--llm-dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="LLM load dtype. auto uses bf16 only when CUDA reports support, otherwise fp16.",
    )
    ap.add_argument(
        "--tokenizer-saved",
        type=Path,
        default=None,
        help="独立 tokenizer 目录（未给则与 --llm 相同）",
    )
    ap.add_argument(
        "--d-llm",
        type=int,
        default=None,
        help="覆盖 Module C d_llm（默认从 AutoConfig 推断）",
    )
    ap.add_argument(
        "--num-llm-layers",
        type=int,
        default=None,
        metavar="N",
        help="覆盖 Module C num_llm_layers（默认从 AutoConfig 推断）",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints/stage3"),
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="最大 epoch 数（早停可能提前结束）")
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="val_loss 连续 N 轮无改善则停（0=关闭；须 --val-jsonl）",
    )
    ap.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=None,
        help="至少训满 N 个 epoch 后才允许早停（默认 config s3_early_stop_min_epochs=2）",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="val 降幅小于该值不算改善（默认 1e-4）",
    )
    ap.add_argument(
        "--s3-lambda-compile",
        type=float,
        default=None,
        help="覆盖 L_compile 权重（0=关闭编译辅助 loss，可加速）",
    )
    ap.add_argument(
        "--scheduled-sampling-start",
        type=float,
        default=None,
        metavar="F",
        help="§7.5.3：每个 epoch 内前 F 比例 batch 100%% teacher forcing（默认 0.6）",
    )
    ap.add_argument(
        "--scheduled-sampling-max-p",
        type=float,
        default=None,
        metavar="P",
        help="§7.5.3：后段 p_sample 上限（默认 0.3）",
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--val-max-samples",
        type=int,
        default=None,
        metavar="N",
        help="验证集仅用前 N 条（与子集冒烟/对齐 train_stage2_binskel）",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="跑一个 batch；加载 --llm 后做一次完整前向",
    )
    ap.add_argument(
        "--val-jsonl",
        type=Path,
        nargs="+",
        default=None,
        help="验证集 jsonl（可选）",
    )
    ap.add_argument("--strict-jsonl", action="store_true")
    ap.add_argument(
        "--max-sft-seq-len",
        type=int,
        default=2048,
        metavar="L",
        help=(
            "Stage3 SFT 序列上限（collate 截断 + 超长样本跳过），默认 2048，与 augment / "
            "24GB 建议及前文设定对齐；设为 0 表示不限制（None）"
        ),
    )
    ap.add_argument(
        "--llm-gradient-checkpointing",
        action="store_true",
        help="对注入后的 HF LLM 调用 gradient_checkpointing_enable（省激活显存）",
    )
    ap.add_argument(
        "--llm-device-map-auto",
        action="store_true",
        help=(
            "以 accelerate ``device_map=auto`` 将 HF 底座切到多张可见 GPU；"
            "A/B encoder 始终在 cuda:0，batch 也 pin 到 cuda:0。"
            "``--dry-run`` 时为单 batch 冒烟；非 dry-run 时为正式 Stage3Trainer 训练（不把整模型 .to 单卡）。"
            "每张卡配额紧时请配 ``--llm-per-gpu-max-gib``（略低于空闲显存）。"
        ),
    )
    ap.add_argument(
        "--llm-per-gpu-max-gib",
        type=float,
        default=None,
        metavar="G",
        help=(
            "与 --llm-device-map-auto 搭配：每张可见 GPU max_memory GiB。"
            "略低于每张空闲显存可促使层跨过更多 GPU（与 scripts/smoke_parallel_llm_forward 一致）。"
        ),
    )
    args = ap.parse_args()
    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    from transformers import AutoConfig, AutoTokenizer

    llm_id = args.llm
    hf_cfg = AutoConfig.from_pretrained(llm_id, trust_remote_code=True)

    cfg = StrucLiftConfig()
    _apply_config_dims(cfg, hf_cfg, args.d_llm, args.num_llm_layers)

    if args.batch_size is not None:
        cfg.training.s3_batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.s3_epochs = args.epochs
    if args.early_stop_patience is not None:
        cfg.training.s3_early_stop_patience = int(args.early_stop_patience)
    if args.early_stop_min_epochs is not None:
        cfg.training.s3_early_stop_min_epochs = int(args.early_stop_min_epochs)
    if args.early_stop_min_delta is not None:
        cfg.training.s3_early_stop_min_delta = float(args.early_stop_min_delta)
    if args.s3_lambda_compile is not None:
        cfg.training.s3_lambda_compile = float(args.s3_lambda_compile)
    if args.scheduled_sampling_start is not None:
        cfg.training.s3_scheduled_sampling_start = float(args.scheduled_sampling_start)
    if args.scheduled_sampling_max_p is not None:
        cfg.training.s3_scheduled_sampling_max_p = float(args.scheduled_sampling_max_p)
    if args.max_sft_seq_len <= 0:
        cfg.training.s3_max_sft_seq_len = None
    else:
        cfg.training.s3_max_sft_seq_len = int(args.max_sft_seq_len)
    if args.llm_gradient_checkpointing:
        cfg.training.llm_gradient_checkpointing = True

    skip_invalid = not args.strict_jsonl

    tok_path_str = (
        str(args.tokenizer_saved.expanduser().resolve())
        if args.tokenizer_saved is not None
        else llm_id
    )
    tokenizer_stage3 = AutoTokenizer.from_pretrained(
        tok_path_str, trust_remote_code=True,
    )
    mft = int(cfg.module_c.max_fill_tokens)
    needed_fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(mft)]
    vocab_map = tokenizer_stage3.get_vocab()
    missing_fill = [t for t in needed_fill_tokens if t not in vocab_map]
    if missing_fill:
        tokenizer_stage3.add_tokens(missing_fill)
    fill_token_ids_cpu = torch.tensor(
        [tokenizer_stage3.convert_tokens_to_ids(t) for t in needed_fill_tokens],
        dtype=torch.long,
    )
    target_vs: int | None = len(tokenizer_stage3)
    pad_id = tokenizer_stage3.pad_token_id
    if pad_id is None:
        pad_id = tokenizer_stage3.eos_token_id
    if pad_id is None:
        pad_id = 0
    pad_id = int(pad_id)
    if args.tokenizer_saved is not None:
        logger.info(
            "使用 --tokenizer-saved: len=%d pad_token_id=%d",
            target_vs or 0,
            pad_id,
        )

    if args.jsonl_list is not None:
        list_path = args.jsonl_list.expanduser().resolve()
        raw = list_path.read_text(encoding="utf-8")
        paths: list[Path] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line).expanduser().resolve())
        if not paths:
            logger.error("--jsonl-list 无有效路径: %s", list_path)
            sys.exit(1)
        jsonl_arg: object = paths[0] if len(paths) == 1 else paths
    else:
        assert args.jsonl is not None
        jsonl_arg = (
            args.jsonl[0].expanduser().resolve()
            if len(args.jsonl) == 1
            else [p.expanduser().resolve() for p in args.jsonl]
        )

    try:
        loader = make_binskel_stage3_dataloader(
            jsonl_arg,
            batch_size=cfg.training.s3_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=getattr(cfg.training, "s3_max_sft_seq_len", None),
            fill_token_ids=fill_token_ids_cpu,
            rebuild_slot_mask_when_missing=True,
        )
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
    if len(loader.dataset) == 0:
        logger.error("无有效样本：检查 jsonl 是否含 alignment_matrix 与 SFT 字段")
        sys.exit(1)

    if args.resume is not None and args.init_from is not None:
        logger.warning("--resume 与 --init-from 同给：忽略 --init-from，从 Stage3 ckpt 续训")

    model = StrucLift(cfg)
    if args.init_from is not None and args.resume is None:
        ckpt_path = args.init_from.expanduser().resolve()
        if not ckpt_path.is_file():
            logger.error("checkpoint 不存在: %s", ckpt_path)
            sys.exit(1)
        try:
            # 必须 CPU 解压：若在 GPU map_location=device，整份 .pt 先占满 VRAM，
            # 再加载 27B + LoRA 必 OOM。load_state_dict 后仍为 CPU，
            # 之后再迁入 GPU：`Stage3Trainer(model.to(device))`，或 `--llm-device-map-auto`
            # 时由 accelerator 占位、仅将 A/B 迁 cuda:0。
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")
        # Stage2 训练器保存为 {"version","model","optimizer",...}，与 scripts/train_stage2_binskel.py 一致须解包。
        if (
            isinstance(state, dict)
            and "model" in state
            and isinstance(state["model"], dict)
        ):
            state = state["model"]
            logger.info("已从 Stage2 封装 ckpt 取出 state[\"model\"] 载入 StrucLift")
        missing, unexpected = load_state_dict_skip_shape_mismatch(
            model, state, log=logger,
        )
        if missing:
            logger.info("load_state_dict 未载入键（示例最多 5 个）: %s", missing[:5])
        if unexpected:
            logger.info("checkpoint 中未使用键（示例最多 5 个）: %s", unexpected[:5])

    if args.dry_run:
        batch = next(iter(loader))
        logger.info("dry-run batch:\n%s", describe_binskel_batch(batch))

        gpu0 = torch.device("cuda:0") if torch.cuda.is_available() else device

        if args.llm_device_map_auto:
            if device.type != "cuda":
                logger.error("--llm-device-map-auto 需要 CUDA 环境（device=cuda）。请设置 --device cuda。")
                sys.exit(1)
            logger.info(
                "dry-run 模式：LLM device_map=auto（多卡切分）；A/B encoder 固定在 cuda:0",
            )
            try:
                llm = _load_hf_llm_autoconfig(
                    cfg,
                    llm_id,
                    device,
                    target_vocab_size=target_vs,
                    device_map_auto=True,
                    llm_max_memory_per_gpu_gib=args.llm_per_gpu_max_gib,
                    llm_dtype=args.llm_dtype,
                )
            except SystemExit:
                raise
            except ImportError as e:
                logger.error("%s", e)
                sys.exit(1)
            except RuntimeError as e:
                logger.error("%s", e)
                sys.exit(1)
            model.decoder.inject_into_llm(llm)
            _align_decoder_adapters_with_llm_shards(model.decoder)
            _maybe_enable_llm_gradient_checkpoint(model, cfg)
            batch = move_binskel_batch_to_device(batch, gpu0)
            _move_struclift_encoders_alignment_to_cuda0(model)
            sw = SlotWeightBuilder()
            fid = fill_token_ids_cpu.to(gpu0)
            slot_weight_mask = batch.get("slot_weight_mask")
            if slot_weight_mask is None:
                slot_weight_mask = sw.build_slot_weights_batched_for_supervised_labels(
                    batch["input_ids"],
                    batch["labels"],
                    batch["slot_types"],
                    cfg.module_c,
                    fill_token_ids=fid,
                )
            model.train()
            model.binary_encoder.eval()
            model.source_encoder.eval()
            model.alignment_module.eval()
        else:
            try:
                llm = _load_hf_llm_autoconfig(
                    cfg,
                    llm_id,
                    device,
                    target_vocab_size=target_vs,
                    llm_dtype=args.llm_dtype,
                )
            except ImportError as e:
                logger.error("%s", e)
                sys.exit(1)
            except RuntimeError as e:
                logger.error("%s", e)
                sys.exit(1)
            model.decoder.inject_into_llm(llm)
            _maybe_enable_llm_gradient_checkpoint(model, cfg)
            batch = move_binskel_batch_to_device(batch, device)
            trainer = Stage3Trainer(
                model,
                cfg,
                device=device,
                fill_token_ids=fill_token_ids_cpu.to(device),
                tokenizer=tokenizer_stage3,
            )
            fid = trainer.fill_token_ids
            slot_weight_mask = batch.get("slot_weight_mask")
            if slot_weight_mask is None:
                slot_weight_mask = trainer.slot_weight_builder.build_slot_weights_batched_for_supervised_labels(
                    batch["input_ids"],
                    batch["labels"],
                    batch["slot_types"],
                    cfg.module_c,
                    fill_token_ids=fid,
                )
            model.train()
            model.binary_encoder.eval()
            model.source_encoder.eval()
            model.alignment_module.eval()

        with torch.amp.autocast(
            "cuda", enabled=cfg.training.fp16 and device.type == "cuda",
        ):
            out = model(
                bin_opcodes=batch["bin_opcodes"],
                bin_operand_types=batch["bin_operand_types"],
                bin_operand_values=batch["bin_operand_values"],
                bin_block_lengths=batch["bin_block_lengths"],
                bin_edge_index=batch["bin_edge_index"],
                bin_edge_type=batch["bin_edge_type"],
                bin_struct_features=batch["bin_struct_features"],
                bin_batch=batch.get("bin_batch"),
                src_token_ids=batch.get("src_token_ids"),
                src_stmt_lengths=batch.get("src_stmt_lengths"),
                src_edge_index=batch.get("src_edge_index"),
                src_edge_type=batch.get("src_edge_type"),
                src_ast_types=batch.get("src_ast_types"),
                src_compound_id=batch.get("src_compound_id"),
                src_batch=batch.get("src_batch"),
                bin_adj=batch.get("adj_bin"),
                src_adj=batch.get("adj_src_gt"),
                M_gt=batch.get("M_gt"),
                run_alignment=True,
                input_ids=batch["input_ids"],
                graph_kv=batch.get("graph_kv"),
                slot_mask=batch.get("slot_mask"),
                labels=batch["labels"],
                slot_weight_mask=slot_weight_mask,
            )
        logger.info(
            "dry-run OK: lm_loss=%s logits=%s",
            out.lm_loss,
            None if out.logits is None else tuple(out.logits.shape),
        )
        return

    train_anchor = device
    if args.llm_device_map_auto:
        if device.type != "cuda":
            logger.error("--llm-device-map-auto 需要 CUDA（请使用 --device cuda 或 cuda:0）")
            sys.exit(1)
        if torch.cuda.device_count() < 2:
            logger.warning(
                "当前仅可见 %d 张 GPU：device_map=auto 可能仍把工作放在少数卡上",
                torch.cuda.device_count(),
            )
        train_anchor = torch.device("cuda", 0)
        _move_struclift_encoders_alignment_to_cuda0(model)
        try:
            llm = _load_hf_llm_autoconfig(
                cfg,
                llm_id,
                device,
                target_vocab_size=target_vs,
                device_map_auto=True,
                llm_max_memory_per_gpu_gib=args.llm_per_gpu_max_gib,
                llm_dtype=args.llm_dtype,
            )
        except SystemExit:
            raise
        except ImportError as e:
            logger.error("需要 transformers / accelerate /（可选）peft。%s", e)
            sys.exit(1)
        except RuntimeError as e:
            logger.error("%s", e)
            sys.exit(1)
        model.decoder.inject_into_llm(llm)
        _align_decoder_adapters_with_llm_shards(model.decoder)
        _maybe_enable_llm_gradient_checkpoint(model, cfg)
        trainer = Stage3Trainer(
            model,
            cfg,
            device=train_anchor,
            fill_token_ids=fill_token_ids_cpu.to(train_anchor),
            tokenizer=tokenizer_stage3,
            decoder_device_map_auto=True,
        )
    else:
        try:
            llm = _load_hf_llm_autoconfig(
                cfg,
                llm_id,
                device,
                target_vocab_size=target_vs,
                llm_dtype=args.llm_dtype,
            )
        except ImportError as e:
            logger.error("需要 transformers /（可选）peft。%s", e)
            sys.exit(1)
        except RuntimeError as e:
            logger.error("%s", e)
            sys.exit(1)
        model.decoder.inject_into_llm(llm)
        _maybe_enable_llm_gradient_checkpoint(model, cfg)
        trainer = Stage3Trainer(
            model,
            cfg,
            device=device,
            fill_token_ids=fill_token_ids_cpu.to(device),
            tokenizer=tokenizer_stage3,
        )

    logger.info(
        "开始 Stage3: 样本数=%d batch_size=%d epochs=%d train_anchor=%s llm_shard=%s llm=%s",
        len(loader.dataset),
        cfg.training.s3_batch_size,
        cfg.training.s3_epochs,
        train_anchor,
        args.llm_device_map_auto,
        llm_id,
    )
    val_loader = None
    val_paths: list[Path] = []
    if args.val_jsonl is not None:
        val_paths = (
            [args.val_jsonl[0].expanduser().resolve()]
            if len(args.val_jsonl) == 1
            else [p.expanduser().resolve() for p in args.val_jsonl]
        )
        val_loader = make_binskel_stage3_dataloader(
            val_paths,
            batch_size=cfg.training.s3_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.val_max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=getattr(cfg.training, "s3_max_sft_seq_len", None),
            fill_token_ids=fill_token_ids_cpu,
            rebuild_slot_mask_when_missing=True,
        )
        logger.info("验证集样本数=%d", len(val_loader.dataset))

    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl_paths: list[str] = []
    if isinstance(jsonl_arg, list):
        train_jsonl_paths = [str(p) for p in jsonl_arg]
    else:
        train_jsonl_paths = [str(jsonl_arg)]
    val_jsonl_paths = [str(p) for p in val_paths]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage_tag": "o0",
        "llm": llm_id,
        "tokenizer_saved": tok_path_str,
        "init_from": (
            str(args.init_from.expanduser().resolve())
            if args.init_from is not None and args.resume is None
            else None
        ),
        "train_jsonl": train_jsonl_paths,
        "val_jsonl": val_jsonl_paths,
        "save_dir": str(save_dir),
        "hyperparams": {
            "epochs_max": cfg.training.s3_epochs,
            "batch_size": cfg.training.s3_batch_size,
            "max_sft_seq_len": cfg.training.s3_max_sft_seq_len,
            "s3_lambda_compile": cfg.training.s3_lambda_compile,
            "early_stop_patience": cfg.training.s3_early_stop_patience,
            "early_stop_min_epochs": cfg.training.s3_early_stop_min_epochs,
            "early_stop_min_delta": cfg.training.s3_early_stop_min_delta,
            "scheduled_sampling_start": cfg.training.s3_scheduled_sampling_start,
            "scheduled_sampling_max_p": cfg.training.s3_scheduled_sampling_max_p,
        },
    }
    (save_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("已写入续训清单: %s", save_dir / "run_manifest.json")

    start_epoch = 1
    initial_best = float("inf")
    initial_no_improve = 0
    initial_global_step = 0
    initial_total_train_steps: int | None = None
    if args.resume is not None:
        rp = args.resume.expanduser().resolve()
        if not rp.is_file():
            logger.error("--resume 文件不存在: %s", rp)
            sys.exit(1)
        ck = trainer.load_full_checkpoint(str(rp))
        start_epoch = int(ck["start_epoch"])
        initial_best = float(ck["best_loss"])
        initial_no_improve = int(ck.get("epochs_without_improve", 0))
        initial_global_step = int(ck.get("global_step", 0))
        tts = ck.get("total_train_steps")
        if tts is not None:
            initial_total_train_steps = int(tts)

    trainer.train(
        loader,
        val_loader=val_loader,
        save_dir=str(save_dir),
        start_epoch=start_epoch,
        initial_best_loss=initial_best,
        initial_epochs_without_improve=initial_no_improve,
        initial_global_step=initial_global_step,
        initial_total_train_steps=initial_total_train_steps,
    )


if __name__ == "__main__":
    main()
