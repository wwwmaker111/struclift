#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage3 推理（eval 前向）性能探针 — 不显式修改 struclift 训练核心代码。

对若干条含 SFT 字段的 binskel jsonl：

1. ``torch.load`` Stage2 ckpt：**默认 ``map_location="cpu"``**，可避免反序列化时直接把张量铺在
   CUDA 上导致 ``cudaErrorDevicesUnavailable``（GPU 忙碌/不可用/已无空闲显存时常见）。
2. 加载 HF LLM → ``inject_into_llm`` → 对数据集前向（no_grad）。
3. 打印「端到端还原」耗时（装载 + N 样本推理）与各阶段占总时间的占比。

示例::

    CUDA_VISIBLE_DEVICES=3,4 python scripts/profile_stage3_infer.py \\
      --jsonl ... --init-from ... --llm .../qwen3.6_27B --tokenizer-saved ... \\
      --max-samples 100 --batch-size 1 --max-sft-seq-len 2048 \\
      --llm-device-map-auto --llm-per-gpu-max-gib 38
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    make_binskel_stage3_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.module_c import SlotWeightBuilder
from struclift.models.struclift import StrucLift, batched_graph_kv_from_nodes
from struclift.utils.state_dict_compat import load_state_dict_skip_shape_mismatch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _cuda_sync_visible_gpus() -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(torch.device("cuda", i))


def _load_train_stage3_autoconfig_module():
    """执行 ``train_stage3_binskel_autoconfig.py`` 获取 ``_apply_config_dims`` /
    ``_load_hf_llm_autoconfig`` 等多卡加载辅助（不向 sys.path 安装包）。"""
    path = SCRIPT_DIR / "train_stage3_binskel_autoconfig.py"
    spec = importlib.util.spec_from_file_location("_t3_autoconfig", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_torch_device(preferred: Optional[str]) -> torch.device:
    """将 ``cuda`` 规范成带 index 的 ``device``；不显式 ``set_device``，避免个别驱动/版本在
    上下文未建立时于 ``set_device`` 处误报 ``busy/unavailable``。"""
    if preferred is None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        dev = torch.device("cuda", 0)
    else:
        dev = torch.device(preferred.strip())
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            logger.error("指定了 CUDA 设备但 torch.cuda.is_available() 为 False")
            raise SystemExit(1)
        idx = 0 if dev.index is None else int(dev.index)
        dev = torch.device("cuda", idx)
    return dev


def _log_cuda_health(dev: torch.device) -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logger.info("[cuda] CUDA_VISIBLE_DEVICES=%s", vis or "(unset)")
    logger.info("[cuda] using torch.device(%r, index=%s)", str(dev), dev.index)
    if dev.type != "cuda":
        return
    idx = 0 if dev.index is None else int(dev.index)
    try:
        free_b, total_b = torch.cuda.mem_get_info(idx)
        logger.info(
            "[cuda] mem_get_info cuda:%d  free %.2f GiB / total %.2f GiB",
            idx,
            free_b / (1024**3),
            total_b / (1024**3),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cuda] mem_get_info failed: %s", exc)
    props = torch.cuda.get_device_properties(idx)
    logger.info("[cuda] name=%s capability=%d.%d", props.name, props.major, props.minor)


def _warmup_cuda_peer_alloc(dev: torch.device) -> None:
    """在搬迁大模块前做小分配；此处若失败多半是驱动/配额/上下文，而不是典型 OOM。"""
    idx = 0 if dev.index is None else int(dev.index)
    try:
        t = torch.empty((4096,), dtype=torch.uint8, device=torch.device("cuda", idx))
        torch.cuda.synchronize()
        del t
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[cuda] 试分配 cuda:%d 上 4KiB tensor 失败（%s）。"
            "这通常「不是」单纯搬 StrucLift 太大（真正 OOM 多在更大连续块失败且信息不同）。"
            "请逐项排查：nvidia-smi 是否与实际作业绑卡一致、cgroup/Slurm GPU 只允许部分卡、"
            "是否需单卡独占 ``export CUDA_VISIBLE_DEVICES=1``、或上一轮 CUDA 报错后重启 shell。"
            "\n底层错误: %s",
            idx,
            type(exc).__name__,
            exc,
        )
        raise SystemExit(2) from exc


def _timed(label: str, fn: Callable[[], Any], timings: Dict[str, float]) -> Any:
    t0 = time.perf_counter()
    out = fn()
    timings[label] = timings.get(label, 0.0) + (time.perf_counter() - t0)
    return out


def _load_ckpt_state(path: Path, map_location: str | torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_llm_pretrained_source(llm_raw: str) -> str:
    """供 ``AutoConfig`` / ``from_pretrained`` 使用：若为本地 HF 权重目录则规范为绝对路径，否则视作 Hub id。"""
    raw = llm_raw.strip()
    if not raw:
        raise SystemExit("--llm 为空；请传入本地模型目录（含 config.json）或 Hugging Face 仓库 id")

    rl = raw.lower().replace("\\", "/")
    if "qwen_or_hub_id" in rl or "/path/to/" in rl:
        raise SystemExit(
            "错误：`--llm` 仍为文档里的占位符路径。请改为服务器上解压后的 HF 权重目录 "
            "(该目录内应有 config.json)，例如 `/data/.../Qwen2.5-7B-Instruct`；"
            "或直接使用 Hub id，例如 `Qwen/Qwen2.5-7B-Instruct`。"
        )

    cand = Path(raw).expanduser()
    try:
        resolved = cand.resolve(strict=False)
    except OSError:
        resolved = cand

    if resolved.is_dir() and (resolved / "config.json").is_file():
        return str(resolved)

    looks_local = resolved.is_absolute() or raw.startswith(("./", "../", "~/", "~\\"))
    if ":" in raw and len(raw) > 2 and raw[1] == ":":  # Windows 盘符
        looks_local = True
    if looks_local:
        logger.error(
            "本地 `--llm` 路径无效或缺少 config.json: %s",
            resolved,
        )
        raise SystemExit(1)

    return raw


def _load_hf_llm(
    cfg: StrucLiftConfig,
    llm_name_or_path: str,
    device: torch.device,
    *,
    llm_dtype_device: Optional[str],
    target_vocab_size: int | None = None,
) -> torch.nn.Module:
    """与 train_stage3_binskel._load_hf_llm 等价（便于脚本自包含）。"""
    from transformers import AutoModelForCausalLM

    if llm_dtype_device is None:
        lm_dev = device
    else:
        lm_dev = torch.device(llm_dtype_device)
    dtype = torch.bfloat16 if lm_dev.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    if target_vocab_size is not None and target_vocab_size > 0:
        cur = llm.get_input_embeddings().num_embeddings
        if target_vocab_size != cur:
            try:
                llm.resize_token_embeddings(
                    target_vocab_size, mean_resizing=False,
                )
            except TypeError:
                llm.resize_token_embeddings(target_vocab_size)
            logger.info(
                "resize_token_embeddings: %d -> %d",
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
        logger.warning("未安装 peft：冻结 LLM 基座权重")
        for p in llm.parameters():
            p.requires_grad = False

    llm = llm.to(lm_dev)
    return llm


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage3 eval 前向：100 样本级耗时与各阶段占比（独立脚本）",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path, nargs="+")
    src.add_argument("--jsonl-list", type=Path, metavar="PATH.txt")
    ap.add_argument(
        "--init-from",
        type=Path,
        required=True,
        help="Stage2（或完整 StrucLift）checkpoint .pt",
    )
    ap.add_argument(
        "--llm",
        type=str,
        default=None,
        help="HF 模型：本地权重目录（含 config.json）或 Hub id（勿使用文档占位符 /path/to/...）",
    )
    ap.add_argument(
        "--tokenizer-saved",
        type=Path,
        default=None,
        help="augment 保存目录；与 Stage3 tokenizer 对齐",
    )
    ap.add_argument("--max-samples", type=int, default=100, metavar="N")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--strict-jsonl", action="store_true")
    ap.add_argument(
        "--ckpt-map-location",
        type=str,
        default="cpu",
        help='torch.load 的 map_location，默认 cpu（可避免 GPU 不可用时的反序列化失败）',
    )
    ap.add_argument(
        "--llm-on-cpu-first",
        action="store_true",
        help="先把 LLM 权重放在 CPU(float32)，再 model.to(device)；大模型可先省 CUDA 瞬时峰值",
    )
    ap.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="不参与统计的首若干个 batch（预热线程/CUDA/JIT）；默认 1",
    )
    ap.add_argument(
        "--max-sft-seq-len",
        type=int,
        default=2048,
        metavar="L",
        help=(
            "与 Stage3 collate 一致：序列截断/超长跳过；设为 0 表示不限制（None）。默认 2048"
        ),
    )
    ap.add_argument(
        "--d-llm",
        type=int,
        default=None,
        help="覆盖 Module C d_llm（默认由 --llm 的 AutoConfig 推断）",
    )
    ap.add_argument(
        "--num-llm-layers",
        type=int,
        default=None,
        metavar="N",
        help="覆盖 Module C num_llm_layers（默认由 AutoConfig 推断）",
    )
    ap.add_argument(
        "--llm-device-map-auto",
        action="store_true",
        help="与 train_stage3_binskel_autoconfig 一致：LLM accelerate device_map 多卡切开",
    )
    ap.add_argument(
        "--llm-per-gpu-max-gib",
        type=float,
        default=None,
        metavar="G",
        help="multi-GPU LLM max_memory GiB（与 train_stage3 相同语义）",
    )
    ap.add_argument(
        "--llm-gradient-checkpointing",
        action="store_true",
        help="对注入后的 HF LLM 开启 gradient checkpointing（对齐训练开销）",
    )
    args = ap.parse_args()

    if args.device:
        device = _resolve_torch_device(args.device.strip())
    else:
        device = _resolve_torch_device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    timings_setup: Dict[str, float] = {}

    if device.type == "cuda":
        _log_cuda_health(device)
        torch.cuda.empty_cache()
        _warmup_cuda_peer_alloc(device)

    cfg = StrucLiftConfig()
    cfg.training.s3_batch_size = args.batch_size
    llm_id = _resolve_llm_pretrained_source(args.llm or cfg.module_c.llm_name)

    if args.llm_device_map_auto and device.type != "cuda":
        logger.error("--llm-device-map-auto 需要 CUDA device")
        sys.exit(1)

    try:
        t3m = _load_train_stage3_autoconfig_module()
    except Exception as e:  # noqa: BLE001
        logger.error("加载 train_stage3_binskel_autoconfig 模块失败（Module C 维度/多卡 LLM）：%s", e)
        raise SystemExit(1) from e

    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(llm_id, trust_remote_code=True)
    t3m._apply_config_dims(cfg, hf_cfg, args.d_llm, args.num_llm_layers)

    if args.max_sft_seq_len <= 0:
        cfg.training.s3_max_sft_seq_len = None
    else:
        cfg.training.s3_max_sft_seq_len = int(args.max_sft_seq_len)

    from transformers import AutoTokenizer

    tok_path_str = (
        str(args.tokenizer_saved.expanduser().resolve())
        if args.tokenizer_saved is not None
        else llm_id
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path_str, trust_remote_code=True,
    )
    mft = int(cfg.module_c.max_fill_tokens)
    needed_fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(mft)]
    vocab_map = tokenizer.get_vocab()
    missing_fill = [t for t in needed_fill_tokens if t not in vocab_map]
    if missing_fill:
        tokenizer.add_tokens(missing_fill)
    fill_token_ids_cpu = torch.tensor(
        [tokenizer.convert_tokens_to_ids(t) for t in needed_fill_tokens],
        dtype=torch.long,
    )
    target_vs = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    pad_id = int(pad_id or 0)

    if args.jsonl is not None:
        paths = [p.expanduser().resolve() for p in args.jsonl]
    else:
        lines = [
            ln.strip()
            for ln in args.jsonl_list.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        paths = [Path(ln).expanduser().resolve() for ln in lines]

    loader = make_binskel_stage3_dataloader(
        paths,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        skip_invalid=not args.strict_jsonl,
        pin_memory=device.type == "cuda",
        pad_token_id=pad_id,
        max_sft_seq_len=getattr(cfg.training, "s3_max_sft_seq_len", None),
        fill_token_ids=fill_token_ids_cpu,
        rebuild_slot_mask_when_missing=True,
    )
    if len(loader.dataset) == 0:
        logger.error("数据集为空")
        sys.exit(1)

    ckpt_path = args.init_from.expanduser().resolve()
    if not ckpt_path.is_file():
        logger.error("checkpoint 不存在: %s", ckpt_path)
        sys.exit(1)

    ml = args.ckpt_map_location
    if ml == "cpu":
        ml_arg: str | torch.device = "cpu"
    elif ml.lower() == "cuda" or ml.startswith("cuda:"):
        ml_arg = torch.device(ml)
        logger.warning(
            "ckpt map_location=%s：若遇 CUDA busy/unavailable，请改用默认 --ckpt-map-location cpu。",
            ml,
        )
    else:
        ml_arg = ml

    state = _timed(
        "01_torch_load_ckpt",
        lambda: _load_ckpt_state(ckpt_path, ml_arg),
        timings_setup,
    )
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]
        logger.info("已从封装 ckpt 使用 state[\"model\"]（Stage2 trainer 格式）载入 StrucLift")

    model = StrucLift(cfg)
    _timed(
        "02_load_state_dict_cpu",
        lambda: load_state_dict_skip_shape_mismatch(model, state, log=logger),
        timings_setup,
    )

    mmap = bool(args.llm_device_map_auto)
    gpu0 = torch.device("cuda", 0)

    def _timing_sync_infer() -> None:
        """encode/decoder 计时的 GPU 栅栏（多卡时同步所有可见卡）。"""
        if mmap:
            _cuda_sync_visible_gpus()
        else:
            _cuda_sync(device)

    batch_anchor = gpu0 if mmap else device

    if mmap:
        _timed(
            "03_move_encoders_align_cuda0",
            lambda: t3m._move_struclift_encoders_alignment_to_cuda0(model),
            timings_setup,
        )
        _cuda_sync(gpu0)
        llm = _timed(
            "04_load_hf_llm_device_map",
            lambda: t3m._load_hf_llm_autoconfig(
                cfg,
                llm_id,
                device,
                target_vocab_size=target_vs,
                device_map_auto=True,
                llm_max_memory_per_gpu_gib=args.llm_per_gpu_max_gib,
            ),
            timings_setup,
        )
        if args.llm_on_cpu_first:
            logger.warning("已忽略 --llm-on-cpu-first（与 --llm-device-map-auto 冲突）")
    else:
        _timed(
            "03_structlift_to_device",
            lambda: model.to(device),
            timings_setup,
        )
        _cuda_sync(device)

        llm_dtype_dev = (
            "cpu" if args.llm_on_cpu_first and device.type == "cuda" else None
        )
        llm = _timed(
            "04_load_hf_llm",
            lambda: _load_hf_llm(
                cfg,
                llm_id,
                device,
                llm_dtype_device=llm_dtype_dev,
                target_vocab_size=target_vs,
            ),
            timings_setup,
        )
        if args.llm_on_cpu_first and device.type == "cuda":
            _timed(
                "05_llm_to_cuda",
                lambda: llm.to(device),
                timings_setup,
            )
            _cuda_sync(device)

    _timed(
        "06_inject_into_llm",
        lambda: model.decoder.inject_into_llm(llm),
        timings_setup,
    )
    if mmap:
        _timed(
            "06b_align_dec_adapters",
            lambda: t3m._align_decoder_adapters_with_llm_shards(model.decoder),
            timings_setup,
        )

    slot_builder = SlotWeightBuilder()
    fill_ids_dev = fill_token_ids_cpu.to(batch_anchor)

    if mmap and args.llm_gradient_checkpointing:
        inner_llm = getattr(model.decoder, "_llm", None)
        if inner_llm is not None and hasattr(inner_llm, "gradient_checkpointing_enable"):
            inner_llm.gradient_checkpointing_enable()
            logger.info("已启用 LLM gradient_checkpointing_enable（对齐训练开销）")

    accum = defaultdict(float)
    n_batches = 0
    n_samples_forward = 0
    warmup = max(0, args.warmup_batches)

    model.train()
    model.binary_encoder.eval()
    model.source_encoder.eval()
    model.alignment_module.eval()

    use_fp16 = cfg.training.fp16 and device.type == "cuda"

    t_loop_wall = time.perf_counter()

    for bi, batch_cpu in enumerate(loader):
        do_time = bi >= warmup
        t_prev = time.perf_counter()

        batch = move_binskel_batch_to_device(batch_cpu, batch_anchor)
        if do_time:
            _cuda_sync(batch_anchor)
            accum["A_batch_pin_h2d"] += time.perf_counter() - t_prev
            t_prev = time.perf_counter()

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        slot_weight_mask = slot_builder.build_slot_weights_batched_for_supervised_labels(
            input_ids,
            labels,
            batch["slot_types"],
            cfg.module_c,
            fill_token_ids=fill_ids_dev,
        )
        if do_time:
            _cuda_sync(batch_anchor)
            accum["B_slot_weight_build"] += time.perf_counter() - t_prev
            t_prev = time.perf_counter()

        bb = batch.get("bin_batch")

        slot_mask_dyn: torch.Tensor | None = batch.get("slot_mask")

        with torch.no_grad():
            use_ac = use_fp16
            _timing_sync_infer()
            t_seg = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_ac):
                enc_out = model.encode(
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
                )
            _timing_sync_infer()
            if do_time:
                accum["C1_encode_AB"] += time.perf_counter() - t_seg

            t_seg = time.perf_counter()
            graph_kv_dyn = batch.get("graph_kv")
            if graph_kv_dyn is None:
                if bb is None:
                    n_n = enc_out.binary_node_embs.size(0)
                    bb = torch.zeros(
                        n_n,
                        dtype=torch.long,
                        device=enc_out.binary_node_embs.device,
                    )
                with torch.amp.autocast("cuda", enabled=use_ac):
                    graph_kv_dyn = batched_graph_kv_from_nodes(
                        enc_out.binary_node_embs,
                        bb,
                        batch_size=int(input_ids.size(0)),
                    )
            _timing_sync_infer()
            if do_time:
                accum["C2_graph_kv_pack"] += time.perf_counter() - t_seg

            if slot_mask_dyn is None:
                slot_mask_dyn = torch.ones(
                    input_ids.size(0),
                    input_ids.size(1),
                    graph_kv_dyn.size(1),
                    device=input_ids.device,
                    dtype=torch.float32,
                )

            t_seg = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_ac):
                logits, lm_loss = model.decoder(
                    input_ids=input_ids,
                    graph_kv=graph_kv_dyn,
                    slot_mask=slot_mask_dyn,
                    labels=labels,
                    slot_weight_mask=slot_weight_mask,
                )
            _timing_sync_infer()
            if do_time:
                accum["C3_decoder_LLM"] += time.perf_counter() - t_seg
                n_batches += 1
                n_samples_forward += int(input_ids.shape[0])

        if lm_loss is None:
            logger.warning("batch_idx=%d lm_loss=None（仍计入 warmup/推理计数）", bi)

        _ = logits

    loop_wall_elapsed = time.perf_counter() - t_loop_wall
    _timing_sync_infer()

    t_setup_total = sum(timings_setup.values())
    infer_only = sum(accum.values())
    infer_only = infer_only if infer_only > 0 else 1e-9
    e2e = t_setup_total + loop_wall_elapsed

    def _pct(part: float, whole: float) -> float:
        return 100.0 * part / whole if whole > 1e-12 else 0.0

    logger.info("")
    logger.info("========== Stage3 「还原」探针 ==========")
    logger.info(
        "数据集条目≈%d ，warmup_batches=%d ，统计 batch 数=%d ，统计样本条数≈%d",
        len(loader.dataset),
        warmup,
        n_batches,
        n_samples_forward,
    )
    logger.info(
        "max_sft_seq_len=%s batch_size=%s mmap_llm=%s（C1=encode A+B，C2=graph_kv 组装，C3=decoder+LLM）",
        cfg.training.s3_max_sft_seq_len,
        args.batch_size,
        mmap,
    )
    logger.info(
        "墙钟汇总: setup=%.3fs | 推理循环 loop_wall=%.3fs（含 DataLoader/主机空隙） | 端到端 e2e≈%.3fs",
        t_setup_total,
        loop_wall_elapsed,
        e2e,
    )

    logger.info("--- Setup 分段（仅占 setup 之和） ---")
    for k in sorted(timings_setup.keys()):
        v = timings_setup[k]
        logger.info(
            "  %-26s %8.3fs  (%5.1f%% of setup)",
            k,
            v,
            _pct(v, t_setup_total),
        )
    logger.info(
        "  %26s %8.1f%% （占端到端）",
        "[setup vs e2e]",
        _pct(t_setup_total, e2e),
    )

    logger.info("--- 推理循环分段（仅占 A+B+C；不含 dataloader/host 间隙） ---")
    infer_stage_keys = (
        "A_batch_pin_h2d",
        "B_slot_weight_build",
        "C1_encode_AB",
        "C2_graph_kv_pack",
        "C3_decoder_LLM",
    )
    for stage in infer_stage_keys:
        v = accum[stage]
        logger.info(
            "  %-26s 均 %.5fs/batch %8.3fs sum  (%5.1f%% infer)",
            stage,
            v / max(n_batches, 1),
            v,
            _pct(v, infer_only),
        )
    logger.info(
        "  %26s %8.1f%% （占端到端）",
        "[infer A+B+C vs e2e]",
        _pct(infer_only, e2e),
    )
    logger.info(
        "  note: loop_wall=%.3fs 与 infer 分段和之差包含 DataLoader/主机侧与非计时间隙。",
        loop_wall_elapsed,
    )


if __name__ == "__main__":
    main()
