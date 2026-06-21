"""
StrucLift configuration — all hyperparameters consolidated in dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ────────────────────────────── vocabulary sizes ──────────────────────────────
@dataclass
class VocabConfig:
    num_opcodes: int = 1024          # opcode vocabulary (x86_64 ≈ 1000+)
    num_operand_types: int = 12      # reg_general, imm_small, mem_base …
    num_value_buckets: int = 64      # log2-bucketed immediate / reg group
    num_edge_types: int = 5          # cond, uncond, back, call, ret
    num_patterns: int = 10           # structural pattern categories
    num_ast_types: int = 16          # source AST statement types
    max_instructions_per_block: int = 64
    max_blocks_per_function: int = 256
    pad_token_id: int = 0


# ─────────────────────────── Module A  (encoder) ─────────────────────────────
@dataclass
class ModuleAConfig:
    d_model: int = 256               # main hidden dimension
    # instruction transformer
    instr_transformer_layers: int = 2
    instr_transformer_heads: int = 4
    instr_ffn_dim: int = 1024        # 4 × d_model
    instr_dropout: float = 0.1
    # structural features
    struct_feat_dim: int = 12
    # edge-typed GAT
    gat_layers: int = 4
    gat_heads: int = 4
    edge_emb_dim: int = 32
    gat_dropout: float = 0.1
    # subgraph pattern classifier
    pattern_k_hop: int = 3
    # §5.3.2 拓扑向量：边分布 + 回边 + 边界最短路 + 规模 + 中心出发最短路统计 + 子图直径
    pattern_topo_feat_dim: int = 13
    # PMA graph pooling
    pma_heads: int = 4
    pma_num_seeds: int = 1


# ─────────────────────────── Module B  (alignment) ───────────────────────────
@dataclass
class ModuleBConfig:
    d_model: int = 256
    # region alignment
    region_tau: float = 0.07
    region_sinkhorn_eps: float = 0.1
    region_sinkhorn_iter: int = 10
    # SCOT node-level alignment（训练应以论文主表为准，按需改下列默认值）
    # 实现：structure_constrained_ot 中「外层 num_outer 轮：拓扑梯度更新代价 → 内层 Sinkhorn」，
    # 与「多轮外层 + 每轮熵正则 OT」一致；若论文写 ADMM/对偶变量，属叙述符号差异，非本实现名。
    scot_tau: float = 0.07
    scot_eps: float = 0.05
    scot_sinkhorn_iter: int = 20
    scot_outer_iter: int = 5
    scot_lambda_struct: float = 1.0
    scot_gamma_topo: float = 0.5
    # §6.3.7 dummy 列：列边际 ν_∅ = scot_dummy_mass_scale · n_b / n_s（OT 内扩一列，输出仍 [n_b, n_s]）
    scot_use_dummy_sink: bool = True
    scot_dummy_mass_scale: float = 0.1
    # §6.3.7 无 M_gt 时列边际 ν：「soft_similarity」用 cos 相似度列和归一化，近似「按语句对应 BB 规模」；
    # 「uniform」为 1/n_s（仅消融）
    nu_no_gt_mode: str = "soft_similarity"
    # §6.3.1 区域级对齐置信度阈值（低于此值的区域对不进节点级 SCOT）
    region_confidence_threshold: float = 0.1
    # §6.5.3 推理回归头：预测源码语句数 n_s
    src_count_regressor: bool = True
    # cross-attention refinement
    cross_attn_layers: int = 2
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1
    cross_attn_mask_threshold: float = 0.01
    # §6.5：在 forward 中从 P_pat + CFG + M 生成结构骨架（供 Module C / 日志）
    emit_skeleton: bool = True


# ─────────────────────────── Module C  (LLM decoder) ─────────────────────────
@dataclass
class ModuleCConfig:
    llm_name: str = "deepseek-ai/deepseek-coder-6.7b-base"
    d_llm: int = 4096               # LLM hidden dimension
    d_graph: int = 256               # graph embedding dimension
    d_bottle: int = 256              # adapter bottleneck
    adapter_heads: int = 4
    # 须与 --llm 对应 HF config.json 的 num_hidden_layers 一致（如 DeepSeek-V4-Flash 为 43）
    num_llm_layers: int = 43
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    # 与 scripts/augment_binskel_sft.py --max-fill-tokens 一致；用于注册占位符与槽位权重
    max_fill_tokens: int = 512
    # slot weights
    slot_weights: dict = field(default_factory=lambda: {
        "FILL_COND": 2.0,
        "FILL_STMT": 1.5,
        "FILL_EXPR": 1.0,
        "FILL_TYPE": 0.5,
        "FILL_PARAM": 1.0,
    })


# ─────────────────────────── Module D  (RL) ──────────────────────────────────
@dataclass
class ModuleDConfig:
    # reward weights
    alpha_compile: float = 0.2
    beta_struct: float = 0.5
    gamma_semantic: float = 0.3
    # structural reward sub-weights
    struct_type_weight: float = 0.4
    struct_ted_weight: float = 0.4
    struct_depth_weight: float = 0.2
    # GRPO
    num_candidates: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    kl_beta: float = 0.05
    rl_lr: float = 5e-6
    rl_epochs: int = 3
    max_grad_norm: float = 1.0
    # §8.3.2 step 7: optionally fine-tune Module B cross-attention at 1/10 lr
    finetune_module_b_cross_attn: bool = True
    module_b_lr_ratio: float = 0.1


# ──────────────────────── training / global ───────────────────────────────────
@dataclass
class TrainingConfig:
    # stage 1 – contrastive pretraining
    s1_lr: float = 1e-4
    s1_weight_decay: float = 1e-2
    s1_epochs: int = 100
    s1_batch_size: int = 256
    s1_infonce_tau: float = 0.07
    # §5.5 + §9.2:
    #   L₁ = L_pattern_cls + λ_recon · L_neighbor_recon
    #        + λ_graph · L_InfoNCE^graph + λ_region · L_InfoNCE^region
    s1_lambda_recon: float = 1.0
    s1_lambda_graph_align: float = 0.5
    s1_lambda_region: float = 0.5       # §9.2: region-level contrastive
    # §9.2 数据增强 / 负样本策略（默认全部关闭保持向后兼容）
    s1_inst_dropout: float = 0.0        # §9.2: 随机 mask 指令比例（建议 0.15）
    s1_edge_dropout: float = 0.0        # §9.2: 随机删 CFG 边比例（建议 0.10）
    s1_pseudo_neg_mask: bool = True     # §9.2: 伪负过滤（同函数不同编译产物不作为负样本）
    s1_cross_opt_pairs: bool = False    # §9.2: 跨优化显式正对（Dataset 成对 + multi-positive InfoNCE）
    s1_hard_neg_refresh_every: int = 0  # §9.2: 硬负挖掘刷新周期（epoch），0=关闭；建议 3-5
    s1_hard_neg_topk: int = 32          # §9.2: 每个 anchor 取 Top-K 硬负（16-64）
    # 早停（仅 Stage1）：连续若干 epoch 训练 total loss 无「明显」下降则停止（需 s1_epochs 足够大作上限）
    s1_early_stop_patience: int = 0     # 0=关闭，仅用 s1_epochs
    s1_early_stop_min_delta: float = 1e-4
    s1_early_stop_min_epochs: int = 5     # 至少训这么多 epoch 后才允许早停
    # stage 2 – alignment
    s2_lr: float = 1e-4
    s2_weight_decay: float = 1e-2
    s2_epochs: int = 80
    s2_batch_size: int = 64
    s2_lambda_scot: float = 1.0
    s2_lambda_edge: float = 0.5
    s2_lambda_region: float = 0.3
    s2_lambda_ca_align: float = 0.2
    s2_lambda_src_count: float = 0.1   # §6.5.3 回归头 loss 权重
    # §9.3 Stage2：节点级硬负（用当前 BB/语句 embedding 对非 GT 位置做 Top-K，InfoNCE 风格）
    s2_node_hard_neg_enabled: bool = True
    s2_node_hard_neg_topk: int = 32
    s2_lambda_node_hard_neg: float = 0.2
    s2_node_hard_neg_pos_thresh: float = 0.05  # M_gt 高于此视为正（不参与硬负池）
    s2_node_hard_neg_interval: int = 5        # 每多少个 epoch 启用一次该项（0=每个 epoch 都启用）
    # 从「首次启用 node_hn 的 epoch」起，在若干**历元**内线性地将 node_hn 权重乘子从 0 升到 1；0=关闭（与旧版等价的满权重、无升权过程）
    s2_node_hard_neg_warmup_epochs: int = 5
    # Stage2 区域 / 节点硬负 InfoNCE 温度；None 时与 Stage1 一致，使用 s1_infonce_tau
    s2_region_infonce_tau: Optional[float] = None
    s2_node_hn_infonce_tau: Optional[float] = None
    # Stage2 是否对 alignment 前向/损失使用 fp16+GradScaler；False=fp32 更稳（Sinkhorn/多任务下易 nan 梯度）
    s2_fp16: bool = False
    # §9.3 Curriculum：每阶段包含的优化档数随 epoch 递增（需 train_stage2 传多档 jsonl）
    s2_curriculum_epochs_per_stage: int = 10  # 前 N epoch 仅 O0，接下来 N 含 O0+O1，… 序为 O0→O1→O2→O3→Os
    # §9.3 Curriculum 自适应进档（与 s2_curriculum_epochs_per_stage 二选一；启用后后者不再用于切档）
    s2_curriculum_adaptive: bool = False  # 需 --curriculum + --val-jsonl-per-curriculum-stage；每档 min/max epoch + val patience
    s2_curriculum_stage_min_epochs: int = 3   # 本档至少训满后才允许因 patience 进下一档
    s2_curriculum_stage_max_epochs: int = 8   # 本档最多训满即进下一档（或结束训练若已在最后一档）
    s2_curriculum_stage_patience: int = 2     # val total 连续若干 epoch 未改善（且已过 min）则进下一档
    s2_curriculum_stage_min_delta: float = 0.0  # 低于该降幅才算「有提升」，用于数值抖动
    # stage 3 – SFT
    s3_lr: float = 1e-4
    s3_weight_decay: float = 1e-2
    s3_epochs: int = 30
    s3_batch_size: int = 8
    # §7.5.3：按**全局训练步**（非 epoch）— 前 start 比例 100% teacher forcing，
    # 后 (1-start) 比例内 p_sample 从 0 线性增至 max_p。
    s3_scheduled_sampling_start: float = 0.6   # fraction of batches within each epoch
    s3_scheduled_sampling_max_p: float = 0.3
    # Stage3 早停（须 val_loader；0=仅跑满 s3_epochs）
    s3_early_stop_patience: int = 0
    s3_early_stop_min_delta: float = 1e-4
    s3_early_stop_min_epochs: int = 2
    # Stage3/4：SFT 序列超过该长度则跳过样本（None=不限制）；collate 也会截断到该长度作双保险。
    # 24GB + 6.7B+LoRA 建议 2048 或更短；仍 OOM 时可设 llm_load_in_8bit=True（需 bitsandbytes）。
    s3_max_sft_seq_len: Optional[int] = None
    # 将基座 LLM 以 8bit 加载（显著省显存；需 pip install bitsandbytes，且仅 CUDA）
    llm_load_in_8bit: bool = False
    # §9.4: L₃ = L_fill^w + λ₆·L_compile（compile reward 作为辅助 SFT loss）
    # 需 batch 内带 tokenizer（train_stage3 已注入）；无 gcc 时 reward=0，仍可微但失去信号。
    s3_lambda_compile: float = 0.05
    # L_compile 计算方式（与 §7.5/§9.4 对齐；旧实现 ``off`` 仅作日志数值，不反传）：
    #   - ``reward_weighted_ce``（默认）：用 greedy 解码得到 R_b，对每样本 fill loss 做 (1-R_b) 加权；
    #         数学上等价于「编译失败的样本，L_fill 权重更大」，可微，开销 ≈ 1 次解码 + 1 次 gcc。
    #   - ``reinforce``：在 FILL 位置从 logits 采样得到 y_sample，整段解码后跑 compile_reward，
    #         以 (R - baseline) * log π(y_sample) 作为 surrogate（与 Stage4 GRPO 同源，但仅采 1 条）。
    #   - ``off``：兼容旧版，仅累加常量到 loss，不反传梯度（不推荐）。
    s3_compile_loss_mode: str = "reward_weighted_ce"
    # ``reinforce`` 模式的 KL/熵正则与 batch baseline 平滑
    s3_compile_reinforce_entropy_coef: float = 0.0
    s3_compile_reinforce_baseline_momentum: float = 0.9
    # stage 4 – RL  (uses ModuleDConfig)
    seed: int = 42
    num_workers: int = 4
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    # Stage3：对 HF LLM 开启 gradient checkpointing，显著降低显存（略慢）
    llm_gradient_checkpointing: bool = False


# ────────────────────────── convenience constructor ──────────────────────────
@dataclass
class StrucLiftConfig:
    vocab: VocabConfig = field(default_factory=VocabConfig)
    module_a: ModuleAConfig = field(default_factory=ModuleAConfig)
    module_b: ModuleBConfig = field(default_factory=ModuleBConfig)
    module_c: ModuleCConfig = field(default_factory=ModuleCConfig)
    module_d: ModuleDConfig = field(default_factory=ModuleDConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
