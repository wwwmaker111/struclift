"""
Binskel JSONL → PyTorch 张量 batch（与 :class:`Stage1Trainer` / :meth:`StrucLift.encode` 对齐）。

每条 jsonl 为**一个函数**样本；batch 内将多个函数的二进制 CFG / 源码 CFG 做 **PyG 式拼接**，
节点下标按图偏移；``bin_batch`` / ``src_batch`` 标明所属样本下标。
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import random
import threading
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Stage2 encoder cache：离线预计算 Module A 输出（bin_node_embs / pattern_logits /
# bin_graph_emb / src_node_embs / src_graph_emb），训练时直接读盘喂给 alignment_module，
# 跳过 binary_encoder / source_encoder forward 与 SubgraphPatternClassifier。
#
# 与 :func:`scripts/precompute_stage2_encoder_cache.py` 配套；缓存按 shard 切片以避免
# 单文件过大；每个 shard 文件 = list[shard_size] of dict[str, Tensor]，dataset 内部
# LRU 缓存最近 K 个 shard，DataLoader worker 以多进程方式加载（pickle-safe）。
# ════════════════════════════════════════════════════════════════════════════

# cache 字段名（与 manifest 对齐；与 :class:`StrucLift.encode` 的 precomputed_* 入参一一对应）
STAGE2_CACHE_FIELDS: Tuple[str, ...] = (
    "bin_node_embs",
    "pattern_logits",
    "bin_graph_emb",
    "src_node_embs",
    "src_graph_emb",
)
# manifest 文件名常量
STAGE2_CACHE_MANIFEST_NAME: str = "manifest.json"
# 缓存协议版本（schema 变化时 bump，并在加载侧明确兼容窗口）
STAGE2_CACHE_VERSION: int = 1


class Stage2EncoderCache:
    """读 :func:`precompute_stage2_encoder_cache.py` 产出的 shard 缓存目录。

    缓存目录结构::

        <cache_dir>/
            manifest.json          # 见 ``STAGE2_CACHE_MANIFEST_NAME``
            shard_00000000.pt
            shard_00000001.pt
            ...

    每个 ``shard_xxx.pt`` = ``list[shard_size]``，元素为
    ``dict[str, Tensor]``（见 :data:`STAGE2_CACHE_FIELDS`）。``Tensor`` 已 detach 并
    存放在 CPU；DataLoader worker 取出后再随 batch 拷到 GPU。

    Parameters
    ----------
    cache_dir
        与 jsonl 一一对应的缓存目录。
    shard_lru_size
        每个 dataset 实例内最多同时驻留的 shard 数（worker 各自一份，按 LRU 淘汰）。
    """

    def __init__(
        self, cache_dir: Union[str, Path], *, shard_lru_size: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / STAGE2_CACHE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Stage2 encoder cache manifest 不存在: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)
        v = int(self.manifest.get("version", -1))
        if v != STAGE2_CACHE_VERSION:
            raise ValueError(
                f"Stage2 encoder cache 版本不匹配: 读到 {v} ≠ 期望 {STAGE2_CACHE_VERSION}",
            )
        self.n_samples: int = int(self.manifest["n_samples"])
        self.shard_size: int = int(self.manifest["shard_size"])
        self.n_shards: int = int(self.manifest["n_shards"])
        self.fields: Tuple[str, ...] = tuple(self.manifest.get("fields", STAGE2_CACHE_FIELDS))
        self.shard_lru_size = max(1, int(shard_lru_size))
        self._shard_cache: "OrderedDict[int, List[Dict[str, torch.Tensor]]]" = OrderedDict()
        self._lock = threading.Lock()

    # ── DataLoader 多进程：放弃已加载 shard，子 worker 重新按需加载 ────────
    def __getstate__(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("_shard_cache", None)
        d.pop("_lock", None)
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._shard_cache = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self.n_samples

    def _shard_path(self, shard_id: int) -> Path:
        return self.cache_dir / f"shard_{shard_id:08d}.pt"

    def _load_shard(self, shard_id: int) -> List[Dict[str, torch.Tensor]]:
        path = self._shard_path(shard_id)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, list):
            raise RuntimeError(f"Stage2 cache shard 损坏（非 list）: {path}")
        return obj

    def get(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(
                f"Stage2EncoderCache idx={idx} 越界（n_samples={self.n_samples}）"
            )
        shard_id = idx // self.shard_size
        slot = idx % self.shard_size
        with self._lock:
            shard = self._shard_cache.get(shard_id)
            if shard is None:
                shard = self._load_shard(shard_id)
                self._shard_cache[shard_id] = shard
                while len(self._shard_cache) > self.shard_lru_size:
                    self._shard_cache.popitem(last=False)
            else:
                # 命中：移至队尾（最近使用）
                self._shard_cache.move_to_end(shard_id)
            if slot >= len(shard):
                raise RuntimeError(
                    f"Stage2 cache shard {shard_id} slot={slot} 越界（len={len(shard)}）",
                )
            return shard[slot]


class Stage2EncoderCachedDataset(Dataset):
    """包装 :class:`BinSkelJsonlDataset`，在 ``__getitem__`` 时附加预计算 encoder 张量。

    底层数据集的 idx 与 cache 的 idx 一一对应；要求生成 cache 的脚本与本次训练用
    **同一份 jsonl**（``BinSkelJsonlDataset`` 的过滤逻辑相同）。返回 dict 中追加
    形如 ``_cache_<field>`` 的键，由 :func:`collate_binskel_stage2_batch` 收集到
    batch 张量。
    """

    _CACHE_KEY_PREFIX: str = "_cache_"

    def __init__(
        self,
        base: BinSkelJsonlDataset,
        cache: Stage2EncoderCache,
    ) -> None:
        super().__init__()
        # 允许 ``len(base) <= len(cache)``：训练时常用 ``--max-samples`` / benchmark 截前 N 条样本，
        # 此时 base 是 cache 的**前缀**，按 idx 一一对应仍然正确。``len(base) > len(cache)``
        # 才是真正的不匹配（cache 漏算）；保持错误以避免 idx 越界。
        if len(base) > len(cache):
            raise ValueError(
                f"BinSkelJsonlDataset 样本数 {len(base)} 大于 Stage2EncoderCache {len(cache)}; "
                f"cache 可能没覆盖全 jsonl，请重新生成或检查 max_samples",
            )
        self._base = base
        self._cache = cache

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(
        self, idx: int,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        item = self._base.__getitem__(idx)
        # 仅 cross_opt_pairs=False 模式（Stage2 默认）；返回单 dict
        if isinstance(item, dict) and item:
            tensors = self._cache.get(idx)
            for k, v in tensors.items():
                if not isinstance(v, torch.Tensor):
                    continue
                item[self._CACHE_KEY_PREFIX + k] = v
        return item


def _has_source_side(d: Dict[str, Any]) -> bool:
    st = d.get("src_stmt_lengths")
    tok = d.get("src_token_ids")
    return (
        isinstance(st, list)
        and bool(st)
        and isinstance(tok, list)
        and sum(int(x) for x in st) == len(tok)
    )


def _validate_sample(
    d: Dict[str, Any],
    *,
    require_source: bool = False,
    require_alignment: bool = False,
    require_sft: bool = False,
    max_sft_seq_len: Optional[int] = None,
) -> Optional[str]:
    """若样本无效返回错误说明，否则返回 None。"""
    try:
        bl = d.get("bin_block_lengths")
        if not isinstance(bl, list) or not bl:
            return "bin_block_lengths 缺失或空"
        n_bb = len(bl)
        ni = sum(int(x) for x in bl)
        op = d.get("bin_opcodes")
        if not isinstance(op, list) or len(op) != ni:
            return f"len(bin_opcodes) != sum(bin_block_lengths) ({len(op) if isinstance(op, list) else '?'} vs {ni})"
        ot = d.get("bin_operand_types")
        ov = d.get("bin_operand_values")
        if not isinstance(ot, list) or len(ot) != ni * 3:
            return "bin_operand_types 长度应为 n_instr*3"
        if not isinstance(ov, list) or len(ov) != ni * 3:
            return "bin_operand_values 长度应为 n_instr*3"
        pl = d.get("pattern_labels")
        if not isinstance(pl, list) or len(pl) != n_bb:
            return "pattern_labels 与基本块数不一致"
        sf = d.get("bin_struct_features")
        if not isinstance(sf, list) or len(sf) != n_bb:
            return "bin_struct_features 行数与 n_bb 不一致"
        for row in sf:
            if not isinstance(row, list) or len(row) != 12:
                return "bin_struct_features 每行应为 12 维"
        bei = d.get("bin_edge_index")
        bet = d.get("bin_edge_type") or []
        if not (isinstance(bei, list) and len(bei) == 2):
            return "bin_edge_index 应为 [src, dst]"
        if len(bei[0]) != len(bei[1]) or len(bei[0]) != len(bet):
            return "bin_edge_index / bin_edge_type 长度不一致"
        need_src = require_source or require_alignment
        n_s = 0
        if need_src:
            if not _has_source_side(d):
                return "require_source: 缺少或非空 src_stmt_lengths / src_token_ids"
            st = d["src_stmt_lengths"]
            n_s = len(st)
            if not isinstance(d.get("src_ast_types"), list) or len(d["src_ast_types"]) != n_s:
                return "src_ast_types 与语句数不一致"
        if require_alignment:
            M = d.get("alignment_matrix")
            if not isinstance(M, list) or len(M) != n_bb:
                return "require_alignment: alignment_matrix 须为长度 n_bb 的列表"
            for i, row in enumerate(M):
                if not isinstance(row, list) or len(row) != n_s:
                    return f"alignment_matrix 第 {i} 行长度须为 n_stmt={n_s}"
            W = d.get("alignment_confidence")
            if W is not None:
                if not isinstance(W, list) or len(W) != n_bb:
                    return "alignment_confidence 行数须等于 n_bb"
                for i, row in enumerate(W):
                    if not isinstance(row, list) or len(row) != n_s:
                        return f"alignment_confidence 第 {i} 行长度须为 n_stmt={n_s}"
        if require_sft:
            si = d.get("sft_input_ids")
            sl = d.get("sft_labels")
            st = d.get("slot_types")
            if not isinstance(si, list) or not isinstance(sl, list) or len(si) != len(sl):
                return "require_sft: sft_input_ids 与 sft_labels 须为等长整型列表"
            if len(si) == 0:
                return "require_sft: SFT 序列长度不能为 0"
            if max_sft_seq_len is not None and len(si) > max_sft_seq_len:
                return (
                    f"require_sft: sft 序列长度 {len(si)} 超过 max_sft_seq_len={max_sft_seq_len}"
                )
            if not isinstance(st, list):
                return "require_sft: slot_types 须为列表"
    except Exception as e:
        return str(e)
    return None


def _flatten_pair_list(
    samples: Sequence[Union[Dict[str, Any], List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """若样本条目是 list（pair 模式下 ``__getitem__`` 的返回形态），展开为扁平 dict list。

    空 list / 空 dict / 缺字段的条目会被丢弃（与 ``skip_invalid=True`` 行为一致）。
    """
    out: List[Dict[str, Any]] = []
    for item in samples:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict) and sub:
                    out.append(sub)
        elif isinstance(item, dict) and item:
            out.append(item)
    return out


def collate_binskel_batch(
    samples: Sequence[Union[Dict[str, Any], List[Dict[str, Any]]]],
    *,
    inst_dropout_p: float = 0.0,
    edge_dropout_p: float = 0.0,
    add_func_uid: bool = True,
) -> Dict[str, Any]:
    """将多条 jsonl 样本拼成一个 batch 张量字典。

    Stage1（§9.2 + §6.2）：同时为每个样本计算 ``binary_regions`` /
    ``source_regions`` / ``region_pos_mask``（均为 Python 对象列表,随 batch 传到
    训练器），用于区域级 InfoNCE。没有 ``alignment_matrix`` 的样本 mask 为 None。

    §9.2 增强（均为训练期可选，eval/推理请保持 0）：
      * ``inst_dropout_p > 0``：随机将 ``p`` 比例的指令置为 PAD（opcode/operand
        类型/value 全部置 0）——模拟 strip/obfuscation 后的信息丢失。
      * ``edge_dropout_p > 0``：随机删除 ``p`` 比例的 CFG 边（同时删 edge_type）。
        注：区域划分在 dropout 之前完成，所以区域结构保持稳定；只有 GNN / neighbor
        / graph_align 走的那份邻接受影响。
      * ``add_func_uid=True``：在 batch 中追加 ``graph_func_uid`` [B]（同 batch 内
        ``(binary, func_name)`` 相同的样本共享同一 uid），供 ``GraphBinarySource
        ContrastiveLoss`` 做伪负过滤（§9.2 "伪负过滤"）。
    """
    if not samples:
        raise ValueError("samples 为空")

    # pair 模式下可能传入 List[List[dict]]，这里统一展开
    flat_samples: List[Dict[str, Any]] = _flatten_pair_list(samples)
    if not flat_samples:
        raise ValueError("samples 有效条目为空（可能 pair 模式下全部被过滤）")
    samples = flat_samples  # type: ignore[assignment]

    need_src = _has_source_side(samples[0])
    for s in samples[1:]:
        if _has_source_side(s) != need_src:
            raise ValueError(
                "同一 batch 内样本须**全部**含源码侧或**全部**不含，不能混用",
            )

    from struclift.utils.graph_utils import (
        compute_binary_regions_dominator_tree,
        compute_source_regions_compound,
    )
    from struclift.models.module_b import build_region_positive_mask

    bin_opcodes: List[int] = []
    bin_ot: List[int] = []
    bin_ov: List[int] = []
    bin_block_lengths: List[int] = []
    bin_edge_src: List[int] = []
    bin_edge_dst: List[int] = []
    bin_edge_type: List[int] = []
    bin_struct_features: List[List[float]] = []
    pattern_labels: List[int] = []
    pattern_weights: List[float] = []
    bin_batch: List[int] = []

    src_token_ids: List[int] = []
    src_stmt_lengths: List[int] = []
    src_edge_src: List[int] = []
    src_edge_dst: List[int] = []
    src_edge_type: List[int] = []
    src_ast_types: List[int] = []
    src_compound_id: List[int] = []
    src_batch: List[int] = []

    bb_offset = 0
    stmt_offset = 0
    has_pattern_w = True

    binary_regions_list: List[List[Dict[str, Any]]] = []
    source_regions_list: List[List[Dict[str, Any]]] = []
    region_pos_mask_list: List[Optional[torch.Tensor]] = []

    # §9.2 伪负过滤：同 batch 内 (binary, func_name) 相同的样本共享同一 uid
    func_uid_map: Dict[Tuple[str, str], int] = {}
    graph_func_uid: List[int] = []
    # §9.2 硬负挖掘：记录每个样本在 Dataset 里的本地下标，用于 HardNegativeBank lookup
    global_sample_idx: List[int] = []

    for gi, s in enumerate(samples):
        bl = [int(x) for x in s["bin_block_lengths"]]
        n_bb = len(bl)
        bin_block_lengths.extend(bl)
        bin_opcodes.extend(int(x) for x in s["bin_opcodes"])
        bin_ot.extend(int(x) for x in s["bin_operand_types"])
        bin_ov.extend(int(x) for x in s["bin_operand_values"])

        if add_func_uid:
            # §9.2 伪负过滤：用 (source_file, func_name) 作为函数身份键
            # —— source_file 与优化级别/架构无关（而 "binary" 路径含 _o0/_o2 会把
            # 同函数不同编译误判成不同 uid，导致伪负过滤失效）
            src_file = str(s.get("source_file", ""))
            bin_path = str(s.get("binary", ""))
            key = (src_file or bin_path, str(s.get("func_name", "")))
            uid = func_uid_map.setdefault(key, len(func_uid_map))
            graph_func_uid.append(uid)
        # 无论 add_func_uid 与否，都透传 _local_idx（若有）
        li = s.get("_local_idx", -1)
        global_sample_idx.append(int(li) if li is not None else -1)
        for row in s["bin_struct_features"]:
            bin_struct_features.append([float(x) for x in row])
        pattern_labels.extend(int(x) for x in s["pattern_labels"])
        bin_batch.extend([gi] * n_bb)

        bconf = s.get("block_src_confidence")
        if isinstance(bconf, list) and len(bconf) == n_bb:
            for row in bconf:
                if isinstance(row, (list, tuple)) and row:
                    pattern_weights.append(float(sum(float(x) for x in row) / len(row)))
                else:
                    pattern_weights.append(1.0)
        else:
            has_pattern_w = False

        bei = s["bin_edge_index"]
        bet = list(s.get("bin_edge_type") or [])
        for i in range(len(bei[0])):
            bin_edge_src.append(int(bei[0][i]) + bb_offset)
            bin_edge_dst.append(int(bei[1][i]) + bb_offset)
            bin_edge_type.append(int(bet[i]) if i < len(bet) else 0)
        bb_offset += n_bb

        if not need_src:
            continue

        tok = s["src_token_ids"]
        stl = s["src_stmt_lengths"]
        src_token_ids.extend(int(x) for x in tok)
        src_stmt_lengths.extend(int(x) for x in stl)
        n_s = len(stl)
        for x in s["src_ast_types"]:
            src_ast_types.append(int(x))
        sci = s.get("src_compound_id")
        if isinstance(sci, list) and len(sci) == n_s:
            src_compound_id.extend(int(x) for x in sci)
        else:
            src_compound_id.extend([-1] * n_s)
        src_batch.extend([gi] * n_s)

        sei = s.get("src_edge_index") or [[], []]
        setypes = list(s.get("src_edge_type") or [])
        for i in range(len(sei[0])):
            src_edge_src.append(int(sei[0][i]) + stmt_offset)
            src_edge_dst.append(int(sei[1][i]) + stmt_offset)
            src_edge_type.append(int(setypes[i]) if i < len(setypes) else 0)
        stmt_offset += n_s

        # ───── §6.2 区域划分 + §9.2 区域 InfoNCE 正样本 mask（每样本独立算） ─────
        pl_local = torch.tensor(
            [int(x) for x in s["pattern_labels"]], dtype=torch.long,
        )
        bei_local = (
            torch.tensor(
                [[int(x) for x in bei[0]], [int(x) for x in bei[1]]],
                dtype=torch.long,
            )
            if len(bei[0]) > 0
            else torch.zeros(2, 0, dtype=torch.long)
        )
        bin_regs = compute_binary_regions_dominator_tree(pl_local, bei_local, n_bb)

        at_local = torch.tensor(
            [int(x) for x in s["src_ast_types"]], dtype=torch.long,
        )
        cid_raw = s.get("src_compound_id")
        cid_local = torch.tensor(
            [int(x) for x in cid_raw] if isinstance(cid_raw, list) and len(cid_raw) == n_s
            else [-1] * n_s,
            dtype=torch.long,
        )
        sei_local = (
            torch.tensor(
                [[int(x) for x in sei[0]], [int(x) for x in sei[1]]],
                dtype=torch.long,
            )
            if len(sei[0]) > 0
            else torch.zeros(2, 0, dtype=torch.long)
        )
        src_regs = compute_source_regions_compound(at_local, sei_local, n_s, cid_local)

        M_raw = s.get("alignment_matrix")
        if (
            isinstance(M_raw, list)
            and len(M_raw) == n_bb
            and len(bin_regs) >= 2
            and len(src_regs) >= 2
        ):
            M_t = torch.tensor(
                [[float(x) for x in row] for row in M_raw], dtype=torch.float32,
            )
            pos_mask = build_region_positive_mask(M_t, bin_regs, src_regs)
        else:
            pos_mask = None

        binary_regions_list.append(bin_regs)
        source_regions_list.append(src_regs)
        region_pos_mask_list.append(pos_mask)

    n_instr = len(bin_opcodes)
    out: Dict[str, torch.Tensor] = {
        "bin_opcodes": torch.tensor(bin_opcodes, dtype=torch.long),
        "bin_operand_types": torch.tensor(bin_ot, dtype=torch.long).view(n_instr, 3),
        "bin_operand_values": torch.tensor(bin_ov, dtype=torch.long).view(n_instr, 3),
        "bin_block_lengths": torch.tensor(bin_block_lengths, dtype=torch.long),
        "bin_edge_index": torch.zeros(2, 0, dtype=torch.long)
        if not bin_edge_src
        else torch.tensor([bin_edge_src, bin_edge_dst], dtype=torch.long),
        "bin_edge_type": torch.tensor(bin_edge_type, dtype=torch.long),
        "bin_struct_features": torch.tensor(bin_struct_features, dtype=torch.float32),
        "pattern_labels": torch.tensor(pattern_labels, dtype=torch.long),
        "bin_batch": torch.tensor(bin_batch, dtype=torch.long),
    }
    if has_pattern_w and len(pattern_weights) == len(pattern_labels):
        out["pattern_weights"] = torch.tensor(pattern_weights, dtype=torch.float32)

    # ───── §9.2 数据增强（仅训练期开启；区域划分已经用完整图算好，这里只动 GNN 输入）─────
    if inst_dropout_p > 0.0 and out["bin_opcodes"].numel() > 0:
        n_instr_t = out["bin_opcodes"].size(0)
        drop = torch.rand(n_instr_t) < float(inst_dropout_p)
        if drop.any():
            out["bin_opcodes"] = out["bin_opcodes"].masked_fill(drop, 0)
            # operand_{types,values} 形状 [n_instr, 3]
            out["bin_operand_types"] = out["bin_operand_types"].masked_fill(
                drop.unsqueeze(-1), 0,
            )
            out["bin_operand_values"] = out["bin_operand_values"].masked_fill(
                drop.unsqueeze(-1), 0,
            )

    if edge_dropout_p > 0.0 and out["bin_edge_index"].size(1) > 0:
        n_edge_t = out["bin_edge_index"].size(1)
        keep = torch.rand(n_edge_t) >= float(edge_dropout_p)
        if not bool(keep.all()):
            out["bin_edge_index"] = out["bin_edge_index"][:, keep]
            out["bin_edge_type"] = out["bin_edge_type"][keep]

    if add_func_uid and graph_func_uid:
        out["graph_func_uid"] = torch.tensor(graph_func_uid, dtype=torch.long)

    if global_sample_idx:
        out["global_sample_idx"] = torch.tensor(global_sample_idx, dtype=torch.long)

    if need_src:
        out["src_token_ids"] = torch.tensor(src_token_ids, dtype=torch.long)
        out["src_stmt_lengths"] = torch.tensor(src_stmt_lengths, dtype=torch.long)
        out["src_edge_index"] = (
            torch.zeros(2, 0, dtype=torch.long)
            if not src_edge_src
            else torch.tensor([src_edge_src, src_edge_dst], dtype=torch.long)
        )
        out["src_edge_type"] = torch.tensor(src_edge_type, dtype=torch.long)
        out["src_ast_types"] = torch.tensor(src_ast_types, dtype=torch.long)
        out["src_compound_id"] = torch.tensor(src_compound_id, dtype=torch.long)
        out["src_batch"] = torch.tensor(src_batch, dtype=torch.long)
        # §6.2 / §9.2 区域级对比所需（Python 对象，不走 pin_memory）
        out["binary_regions_list"] = binary_regions_list
        out["source_regions_list"] = source_regions_list
        out["region_pos_mask_list"] = region_pos_mask_list
    return out


def collate_binskel_stage2_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stage2：在 :func:`collate_binskel_batch` 基础上拼接 ``M_gt``、邻接与置信度。

    与 :class:`struclift.training.stage2_alignment.Stage2Trainer` 及提案 §6.6 一致：

    - ``M_gt`` : [B, max_nb, max_ns]
    - ``alignment_confidence`` : [B, max_nb, max_ns]（缺失时置 1）
    - ``adj_bin`` : [B, max_nb, max_nb]（有向 CFG，存在边则为 1）
    - ``adj_src_gt`` : [B, max_ns, max_ns]

    若样本来自 :class:`Stage2EncoderCachedDataset`，会附带 ``_cache_<field>`` 键
    （见 :data:`STAGE2_CACHE_FIELDS`）；本函数会按 PyG batch 的拼接顺序 ``torch.cat``
    成 ``cache_<field>``，与 :meth:`StrucLift.encode` 的 ``precomputed_*`` 入参一一对应。
    """
    if not samples:
        raise ValueError("samples 为空")
    base = collate_binskel_batch(samples)
    if "src_batch" not in base:
        raise ValueError("Stage2 需要每条样本含源码侧（与 collate_binskel_batch require_source 一致）")

    B = len(samples)
    max_nb = max(len(s["bin_block_lengths"]) for s in samples)
    max_ns = max(len(s["src_stmt_lengths"]) for s in samples)

    M_gt = torch.zeros(B, max_nb, max_ns, dtype=torch.float32)
    conf = torch.ones(B, max_nb, max_ns, dtype=torch.float32)
    adj_bin = torch.zeros(B, max_nb, max_nb, dtype=torch.float32)
    adj_src_gt = torch.zeros(B, max_ns, max_ns, dtype=torch.float32)

    for gi, s in enumerate(samples):
        n_bb = len(s["bin_block_lengths"])
        n_s = len(s["src_stmt_lengths"])
        M = s["alignment_matrix"]
        M_gt[gi, :n_bb, :n_s] = torch.tensor(
            [[float(x) for x in row] for row in M], dtype=torch.float32,
        )
        W = s.get("alignment_confidence")
        if isinstance(W, list) and len(W) == n_bb:
            conf[gi, :n_bb, :n_s] = torch.tensor(
                [[float(x) for x in row] for row in W], dtype=torch.float32,
            )

        bei = s["bin_edge_index"]
        for k in range(len(bei[0])):
            u, v = int(bei[0][k]), int(bei[1][k])
            if 0 <= u < n_bb and 0 <= v < n_bb:
                adj_bin[gi, u, v] = 1.0

        sei = s.get("src_edge_index") or [[], []]
        for k in range(len(sei[0])):
            u, v = int(sei[0][k]), int(sei[1][k])
            if 0 <= u < n_s and 0 <= v < n_s:
                adj_src_gt[gi, u, v] = 1.0

    base["M_gt"] = M_gt
    base["alignment_confidence"] = conf
    base["adj_bin"] = adj_bin
    base["adj_src_gt"] = adj_src_gt

    # ── encoder cache 旁路：若全部样本均带 _cache_<field>，按 PyG 顺序拼接 ─────
    # 任一样本缺失 cache 时整 batch 视为非缓存（训练侧将退回 forward 路径）。
    cache_prefix = Stage2EncoderCachedDataset._CACHE_KEY_PREFIX
    has_cache = all(
        all((cache_prefix + f) in s for f in STAGE2_CACHE_FIELDS) for s in samples
    )
    if has_cache:
        for f in STAGE2_CACHE_FIELDS:
            tensors = [s[cache_prefix + f] for s in samples]
            base["cache_" + f] = torch.cat(tensors, dim=0)
    return base


def _infer_stage3_slot_to_bb_mapping(
    sample: Dict[str, Any],
    sft_token_ids: List[int],
    *,
    fill_token_ids: Optional[torch.Tensor],
    fill_lo: int,
) -> Optional[Dict[int, List[int]]]:
    """无 ``sft_fill_positions`` / ``sft_slot_bb_indices`` 时，由 ``pattern_labels`` + CFG
    重建骨架槽位，并在 ``sft_token_ids`` 中用 FILL id 定位占位符，构造 §7.3.2 映射。

    与 ``augment_binskel_sft.build_skeleton_for_binary_cfg`` 语义一致；若 FILL 个数与槽位数不一致则返回 ``None``。
    """
    from struclift.utils.skeleton import build_skeleton_for_binary_cfg

    pl = sample.get("pattern_labels")
    bei = sample.get("bin_edge_index")
    bl = sample.get("bin_block_lengths")
    if not isinstance(pl, list) or not pl:
        return None
    if not isinstance(bei, list) or len(bei) != 2:
        return None
    if not isinstance(bl, list) or len(bl) != len(pl):
        return None
    try:
        n_bb = len(bl)
        pl_t = torch.tensor([int(x) for x in pl], dtype=torch.long)
        ei = torch.tensor(
            [[int(x) for x in bei[0]], [int(x) for x in bei[1]]],
            dtype=torch.long,
        )
        skel = build_skeleton_for_binary_cfg(pl_t, ei, n_bb)
        n_slot = len(skel.slots)
        if n_slot == 0:
            return None
        row = torch.tensor(sft_token_ids, dtype=torch.long)
        if fill_token_ids is not None and fill_token_ids.numel() > 0:
            fid = fill_token_ids.detach().cpu().long().flatten()
            positions = torch.where(torch.isin(row, fid))[0].tolist()
        else:
            positions = torch.where(row >= int(fill_lo))[0].tolist()
        if len(positions) != n_slot:
            return None
        mapping: Dict[int, List[int]] = {}
        for k, p in enumerate(positions):
            raw_bbs = skel.slots[k].aligned_bb_indices
            if not raw_bbs:
                continue
            good = [int(x) for x in raw_bbs if 0 <= int(x) < n_bb]
            if good:
                mapping[int(p)] = good
        return mapping if mapping else None
    except Exception:
        return None


def collate_binskel_stage3_batch(
    samples: List[Dict[str, Any]],
    *,
    pad_token_id: int = 0,
    max_sft_seq_len: Optional[int] = None,
    fill_token_ids: Optional[torch.Tensor] = None,
    rebuild_slot_mask_when_missing: bool = True,
    bin_only: bool = False,
) -> Dict[str, Any]:
    """Stage3 SFT：在 Stage2 collate 上增加 ``input_ids`` / ``labels`` / ``slot_types``。

    若样本含 ``sft_fill_positions`` / ``sft_slot_bb_indices``（``augment_binskel_sft.py`` 生成），
    则按提案 §7.3.2 构造 ``slot_mask``。

    若缺失上述字段且 ``rebuild_slot_mask_when_missing=True``，则在 collate 内用
    ``pattern_labels`` + ``bin_edge_index`` 调用 ``build_skeleton_for_binary_cfg`` 推断槽位
    ↔ BB，并结合 ``fill_token_ids``（推荐）或 ``id>=32000`` 定位 FILL 位置，生成等价 ``slot_mask``，
    **无需重写 jsonl**。

    若仍无法构造则 ``slot_mask`` 退化为全 1。

    ``bin_only=True``：仅拼二进制侧（部署推理，无 ``src_*`` / ``M_gt``），走
    :func:`collate_binskel_batch` 而非 Stage2。

    ``max_sft_seq_len``：若设置，对每条样本的 ``sft_*`` / ``slot_types`` 做**前缀保留**截断
    （与数据集索引过滤一致），避免未同步旧代码时仍拼出超长 batch。
    """
    from struclift.models.module_c import StructureConditionedDecoder

    if bin_only:
        base = collate_binskel_batch(samples)
    else:
        base = collate_binskel_stage2_batch(samples)
    B = len(samples)
    trimmed: List[Tuple[List[Any], List[Any], List[Any]]] = []
    for s in samples:
        si = list(s["sft_input_ids"])
        sl = list(s["sft_labels"])
        st = list(s["slot_types"])
        if len(si) != len(sl):
            n = min(len(si), len(sl))
            si, sl = si[:n], sl[:n]
        if max_sft_seq_len is not None and len(si) > max_sft_seq_len:
            si = si[:max_sft_seq_len]
            sl = sl[:max_sft_seq_len]
            # slot_types 按「槽」计数，不能按 token 长度截断
        trimmed.append((si, sl, st))
    max_L = max(len(t[0]) for t in trimmed)
    max_nb = max(len(s["bin_block_lengths"]) for s in samples)
    fill_lo = int(StructureConditionedDecoder.FILL_TOKEN_ID_RANGE_START)

    input_ids = torch.full((B, max_L), pad_token_id, dtype=torch.long)
    labels = torch.full((B, max_L), -100, dtype=torch.long)
    slot_types_batch: List[List[str]] = []
    slot_mask = torch.ones(B, max_L, max_nb, dtype=torch.float32)
    slot_output_spans_batch: List[List[List[int]]] = []
    slot_weight_mask = torch.zeros(B, max_L, dtype=torch.float32)
    has_slot_output_spans = False

    for gi, (si, sl, st) in enumerate(trimmed):
        L = len(si)
        input_ids[gi, :L] = torch.tensor(si, dtype=torch.long)
        labels[gi, :L] = torch.tensor(sl, dtype=torch.long)
        slot_types_batch.append([str(x) for x in st])

        s = samples[gi]
        pos_list = s.get("sft_fill_positions")
        bb_lists = s.get("sft_slot_bb_indices")
        span_lists = s.get("sft_slot_output_spans")
        spans_for_batch: List[List[int]] = []
        mapping: Optional[Dict[int, List[int]]] = None
        if (
            isinstance(pos_list, list)
            and isinstance(bb_lists, list)
            and len(pos_list) == len(bb_lists)
        ):
            mapping = {}
            n_bb_i = len(s["bin_block_lengths"])
            for p_raw, bbs in zip(pos_list, bb_lists):
                p = int(p_raw)
                if p >= L:
                    continue
                if not isinstance(bbs, list):
                    continue
                good = [
                    int(x) for x in bbs
                    if isinstance(x, (int, float)) and 0 <= int(x) < min(n_bb_i, max_nb)
                ]
                if good:
                    mapping[p] = good

        if not mapping and rebuild_slot_mask_when_missing:
            inferred = _infer_stage3_slot_to_bb_mapping(
                s, si, fill_token_ids=fill_token_ids, fill_lo=fill_lo,
            )
            if inferred:
                mapping = inferred

        if mapping:
            row = StructureConditionedDecoder.build_slot_mask(
                input_ids[gi : gi + 1, :L],
                [mapping],
                max_nb,
                fill_token_range_start=fill_lo,
                fill_token_ids=fill_token_ids,
            )
            pad_l = max_L - L
            if pad_l > 0:
                tail = torch.ones(1, pad_l, max_nb, dtype=torch.float32)
                row = torch.cat([row, tail], dim=1)
            slot_mask[gi : gi + 1] = row

        if (
            isinstance(span_lists, list)
            and isinstance(bb_lists, list)
            and len(span_lists) == len(bb_lists)
        ):
            n_bb_i = len(s["bin_block_lengths"])
            for span_raw, bbs in zip(span_lists, bb_lists):
                if not (
                    isinstance(span_raw, list)
                    and len(span_raw) == 2
                    and all(isinstance(x, (int, float)) for x in span_raw)
                ):
                    spans_for_batch.append([0, 0])
                    continue
                start = max(0, min(int(span_raw[0]), L))
                end = max(start, min(int(span_raw[1]), L))
                spans_for_batch.append([start, end])
                if start >= end:
                    continue
                good = [
                    int(x) for x in (bbs if isinstance(bbs, list) else [])
                    if isinstance(x, (int, float)) and 0 <= int(x) < min(n_bb_i, max_nb)
                ]
                if not good:
                    continue
                has_slot_output_spans = True
                slot_weight_mask[gi, start:end] = (labels[gi, start:end] != -100).float()

                # Labels at positions [start, end) are predicted from logits at
                # positions [start - 1, end - 1). Route those predictor states to
                # the same BBs as the slot so V10 output spans keep BB grounding.
                pred_start = max(start - 1, 0)
                pred_end = max(end - 1, pred_start)
                if pred_start < pred_end:
                    slot_mask[gi, pred_start:pred_end, :] = 0.0
                    for bb_idx in good:
                        slot_mask[gi, pred_start:pred_end, bb_idx] = 1.0

        slot_output_spans_batch.append(spans_for_batch)

    base["input_ids"] = input_ids
    base["labels"] = labels
    base["slot_types"] = slot_types_batch
    base["slot_mask"] = slot_mask
    if has_slot_output_spans:
        base["slot_output_spans"] = slot_output_spans_batch
        base["slot_weight_mask"] = slot_weight_mask
    return base


def collate_binskel_stage3_with_skip(
    batch: List[Dict[str, Any]],
    *,
    pad_token_id: int = 0,
    max_sft_seq_len: Optional[int] = None,
    fill_token_ids: Optional[torch.Tensor] = None,
    rebuild_slot_mask_when_missing: bool = True,
    bin_only: bool = False,
) -> Dict[str, Any]:
    good = [b for b in batch if b]
    if not good:
        raise RuntimeError("本 batch 无有效样本")
    return collate_binskel_stage3_batch(
        good,
        pad_token_id=pad_token_id,
        max_sft_seq_len=max_sft_seq_len,
        fill_token_ids=fill_token_ids,
        rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
        bin_only=bin_only,
    )


# 与 :data:`VocabConfig.num_patterns` 默认一致（RL 结构奖励用）
_DEFAULT_NUM_PATTERNS: int = 10


def collate_binskel_stage4_batch(
    samples: List[Dict[str, Any]],
    *,
    pad_token_id: int = 0,
    num_patterns: int = _DEFAULT_NUM_PATTERNS,
    max_sft_seq_len: Optional[int] = None,
    fill_token_ids: Optional[torch.Tensor] = None,
    rebuild_slot_mask_when_missing: bool = True,
) -> Dict[str, Any]:
    """Stage4 GRPO：在 Stage3 collate 上增加结构奖励所需的可选字段。"""
    base = collate_binskel_stage3_batch(
        samples,
        pad_token_id=pad_token_id,
        max_sft_seq_len=max_sft_seq_len,
        fill_token_ids=fill_token_ids,
        rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
    )
    B = len(samples)
    C = num_patterns
    bin_pattern_dist = torch.full((B, C), 1.0 / C, dtype=torch.float32)
    bin_cst_batch: List[List[str]] = []
    depths: List[int] = []
    for gi, s in enumerate(samples):
        pd = s.get("bin_pattern_dist")
        if isinstance(pd, list) and len(pd) == C:
            row = torch.tensor([float(x) for x in pd], dtype=torch.float32)
            row = row.clamp(min=1e-8)
            bin_pattern_dist[gi] = row / row.sum()
        cst = s.get("bin_cst_serialised")
        bin_cst_batch.append([str(x) for x in cst] if isinstance(cst, list) else [])
        d0 = s.get("bin_nesting_depth")
        depths.append(int(d0) if isinstance(d0, int) else 0)
    base["bin_pattern_dist"] = bin_pattern_dist
    base["bin_cst_serialised_batch"] = bin_cst_batch
    base["bin_nesting_depth"] = torch.tensor(depths, dtype=torch.long)
    return base


def collate_binskel_stage4_with_skip(
    batch: List[Dict[str, Any]],
    *,
    pad_token_id: int = 0,
    num_patterns: int = _DEFAULT_NUM_PATTERNS,
    max_sft_seq_len: Optional[int] = None,
    fill_token_ids: Optional[torch.Tensor] = None,
    rebuild_slot_mask_when_missing: bool = True,
) -> Dict[str, Any]:
    good = [b for b in batch if b]
    if not good:
        raise RuntimeError("本 batch 无有效样本")
    return collate_binskel_stage4_batch(
        good,
        pad_token_id=pad_token_id,
        num_patterns=num_patterns,
        max_sft_seq_len=max_sft_seq_len,
        fill_token_ids=fill_token_ids,
        rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
    )


def collate_binskel_stage2_with_skip(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """``DataLoader`` collate：过滤空样本后调用 :func:`collate_binskel_stage2_batch`。"""
    good = [b for b in batch if b]
    if not good:
        raise RuntimeError("本 batch 无有效样本")
    return collate_binskel_stage2_batch(good)


def collate_binskel_with_skip(batch: List[Any]) -> Dict[str, torch.Tensor]:
    """``DataLoader(collate_fn=...)``：过滤 ``__getitem__`` 返回的空 ``{}``（读盘/校验失败时）。

    支持 pair 模式：若条目为 list（``[anchor, companion]``），会在
    :func:`collate_binskel_batch` 内自动展平。
    """
    good: List[Any] = [b for b in batch if b]
    if not good:
        raise RuntimeError("本 batch 无有效样本")
    return collate_binskel_batch(good)


def _collate_binskel_with_skip_kw(
    batch: List[Any],
    *,
    inst_dropout_p: float = 0.0,
    edge_dropout_p: float = 0.0,
    add_func_uid: bool = True,
) -> Dict[str, Any]:
    """带参数版的 skip-wrapper，配合 ``functools.partial`` 传给 DataLoader。

    必须是**顶层函数**才能在 ``num_workers>0`` 的多进程环境下被 pickle。
    """
    good: List[Any] = [b for b in batch if b]
    if not good:
        raise RuntimeError("本 batch 无有效样本")
    return collate_binskel_batch(
        good,
        inst_dropout_p=inst_dropout_p,
        edge_dropout_p=edge_dropout_p,
        add_func_uid=add_func_uid,
    )


def move_binskel_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> Dict[str, Any]:
    """将 batch 中张量搬到 ``device``（其它键原样保留）。"""
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def describe_binskel_batch(batch: Dict[str, Any]) -> str:
    """调试用：一行摘要各张量形状与 dtype。"""
    parts: List[str] = []
    for k in sorted(batch.keys()):
        v = batch[k]
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}={tuple(v.shape)}/{v.dtype}")
        elif k == "slot_types" and isinstance(v, list):
            parts.append(f"{k}=list[len={len(v)}]")
    return "  " + "\n  ".join(parts)


class BinSkelJsonlDataset(Dataset):
    """按行读取 binskel ``.jsonl``，每条为一个训练样本（一个函数）。

    §9.2 跨优化正样本（``cross_opt_pairs=True``）：
      * 索引阶段**额外**建 ``(source_file, func_name) → [sample_idx]``；
      * ``__getitem__(idx)`` 返回 **[anchor, companion]**，companion 从同函数的其它
        编译产物里随机抽取（无伴侣则退化为 ``[anchor, anchor]``，loss 侧通过 eye
        mask 自动把对角作为自身正对，依赖 dropout 增强带来扰动）。
      * 每个 DataLoader worker / DDP rank 内随机源独立，**无需自定义 Sampler**。
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        skip_invalid: bool = True,
        require_source: bool = False,
        require_alignment: bool = False,
        require_sft: bool = False,
        max_samples: Optional[int] = None,
        max_sft_seq_len: Optional[int] = None,
        cross_opt_pairs: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"jsonl 不存在: {self.path}")
        self.skip_invalid = skip_invalid
        if require_sft:
            require_alignment = True
            require_source = True
        self.require_sft = require_sft
        self.require_source = require_source or require_alignment
        self.require_alignment = require_alignment
        self.max_samples = max_samples
        self._max_sft_seq_len = max_sft_seq_len
        self.cross_opt_pairs = bool(cross_opt_pairs)
        self._line_offsets: List[int] = []
        # §9.2 cross_opt_pairs：(source_file, func_name) → [sample_idx]
        self._func_index: Dict[Tuple[str, str], List[int]] = {}
        # 每个 worker 独立随机源，避免多进程并发下依赖全局 random 的可复现问题
        self._rng: Optional[random.Random] = None
        # DataLoader worker 内复用读路径：优先 mmap 整文件（随机 seek 时页缓存更友好）；可设 STRUCLIFT_NO_JSONL_MMAP=1 关闭
        self._fp: Optional[Any] = None
        self._mm: Optional[Any] = None  # mmap.mmap
        self._use_mmap: bool = os.environ.get("STRUCLIFT_NO_JSONL_MMAP", "").lower() not in (
            "1", "true", "yes",
        )
        self._build_index()

    def __getstate__(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("_fp", None)
        d.pop("_mm", None)
        d.pop("_rng", None)
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fp = None
        self._mm = None
        self._rng = None

    def _ensure_read_handle(self) -> None:
        if self._mm is not None or (not self._use_mmap and self._fp is not None):
            return
        f = self.path.open("rb")
        self._fp = f
        if not self._use_mmap:
            return
        try:
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError:
            self._mm = None

    def _build_index(self) -> None:
        self._line_offsets = []
        self._func_index = {}
        n_ok = 0
        with self.path.open("rb") as f:
            while True:
                start = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    d = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    if not self.skip_invalid:
                        raise
                    logger.debug("跳过 JSON 解析失败 offset=%s", start)
                    continue
                err = _validate_sample(
                    d,
                    require_source=self.require_source,
                    require_alignment=self.require_alignment,
                    require_sft=self.require_sft,
                    max_sft_seq_len=self._max_sft_seq_len,
                )
                if err is not None:
                    if self.skip_invalid:
                        logger.debug("跳过无效样本 offset=%s: %s", start, err)
                        continue
                    raise ValueError(f"无效样本 offset={start}: {err}")
                sample_idx = len(self._line_offsets)
                self._line_offsets.append(start)
                if self.cross_opt_pairs:
                    key = (
                        str(d.get("source_file", "") or d.get("binary", "")),
                        str(d.get("func_name", "")),
                    )
                    self._func_index.setdefault(key, []).append(sample_idx)
                n_ok += 1
                if self.max_samples is not None and n_ok >= self.max_samples:
                    break
        n_funcs = len(self._func_index) if self.cross_opt_pairs else 0
        n_with_companion = (
            sum(1 for v in self._func_index.values() if len(v) >= 2)
            if self.cross_opt_pairs else 0
        )
        logger.info(
            "BinSkelJsonlDataset: %s 有效样本数=%d (require_source=%s require_sft=%s%s)",
            self.path.name,
            len(self._line_offsets),
            self.require_source,
            self.require_sft,
            f" cross_opt_pairs=True funcs={n_funcs} with_companion={n_with_companion}"
            if self.cross_opt_pairs else "",
        )

    def __len__(self) -> int:
        return len(self._line_offsets)

    def _read_one(self, idx: int) -> Dict[str, Any]:
        """从 ``_line_offsets[idx]`` 读并校验单条样本；失败（``skip_invalid``）返回 ``{}``。"""
        start = self._line_offsets[idx]
        try:
            self._ensure_read_handle()
            if self._mm is not None:
                self._mm.seek(start)
                line = self._mm.readline()
            else:
                assert self._fp is not None
                self._fp.seek(start)
                line = self._fp.readline()
            d = json.loads(line.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError, UnicodeError) as e:
            if self.skip_invalid:
                logger.warning("读取/解析失败 idx=%d offset=%s: %s", idx, start, e)
                return {}
            raise
        err = _validate_sample(
            d,
            require_source=self.require_source,
            require_alignment=self.require_alignment,
            require_sft=self.require_sft,
            max_sft_seq_len=self._max_sft_seq_len,
        )
        if err is not None:
            if self.skip_invalid:
                logger.warning("样本无效 idx=%d: %s", idx, err)
                return {}
            raise ValueError(f"无效样本 idx={idx}: {err}")
        return d

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            # 每个进程独立 seed：worker_id + pid + os.urandom 熵；PyTorch DataLoader
            # 的 worker_init_fn 若未配置也能有独立随机
            seed_src = (os.getpid(), id(self))
            self._rng = random.Random(hash(seed_src) & 0xFFFFFFFF)
        return self._rng

    def __getitem__(
        self, idx: int,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        anchor = self._read_one(idx)
        if anchor:
            # §9.2 硬负挖掘：记录样本在当前 Dataset 中的本地下标。多文件
            # ConcatDataset 暂不支持硬负，训练脚本会拒绝启用。
            anchor["_local_idx"] = int(idx)
        if not self.cross_opt_pairs:
            return anchor
        # §9.2 cross_opt_pairs 模式
        if not anchor:
            # anchor 无效：整对丢弃（collate 的 skip 包装会把空值过滤掉）
            return []
        # 找同函数 companion
        key = (
            str(anchor.get("source_file", "") or anchor.get("binary", "")),
            str(anchor.get("func_name", "")),
        )
        pool = self._func_index.get(key, [])
        companion: Dict[str, Any] = {}
        cand_idx = idx
        if len(pool) >= 2:
            rng = self._get_rng()
            for _ in range(5):  # 最多重试 5 次避免读到坏样本
                cand = idx
                while cand == idx:
                    cand = rng.choice(pool)
                companion = self._read_one(cand)
                if companion:
                    cand_idx = cand
                    break
        if not companion:
            # 没有 companion（孤儿函数或连续读坏）：退化为自配对——依靠 §9.2
            # 的 inst/edge dropout 给两份独立扰动，近似 SimCLR 式正对
            companion = dict(anchor)  # 浅拷贝，避免 companion["_local_idx"] 覆盖 anchor
        companion["_local_idx"] = int(cand_idx)
        return [anchor, companion]


class Stage2CurriculumDataset(Dataset):
    """§9.3 Curriculum Learning：先 O0，再逐步并入 O1→O2→O3→Os。

    通过 ``set_stage(k)`` 控制当前训练可见的优化档前缀（``k=0`` 仅 O0 文件，
    ``k=1`` 为 O0+O1 的样本并集，…）。**长度随 stage 变化**，调用方须在
    stage 切换时重建带 ``shuffle=True`` 的 DataLoader。

    课程顺序 **O0 → O1 → O2 → O3 → Os**（与 StrucLift 训练约定一致：先标准优化级，
    最后并入 ``-Os`` 体积优选项）。若 ``paths_by_opt`` 未提供某档，则该档自动跳过。
    """

    OPT_ORDER: Tuple[str, ...] = ("O0", "O1", "O2", "O3", "Os")

    def __init__(
        self,
        paths_by_opt: Dict[str, Path],
        *,
        max_samples: Optional[int] = None,
        skip_invalid: bool = True,
        encoder_cache_dir: Optional[Union[str, Path]] = None,
        encoder_cache_shard_lru: int = 4,
    ) -> None:
        """``encoder_cache_dir`` 非 None 时，每个 jsonl 旁挂一个同名子目录的 cache：

        ::

            <encoder_cache_dir>/<jsonl_filename_stem>/manifest.json + shard_*.pt

        子目录名取自 jsonl 路径的 ``Path.stem``（不含 ``.jsonl`` 后缀）；
        要求每档都已通过 :file:`scripts/precompute_stage2_encoder_cache.py` 预生成。
        """
        super().__init__()
        self.datasets: List[Dataset] = []
        self._labels: List[str] = []
        cache_root: Optional[Path] = (
            Path(encoder_cache_dir).expanduser().resolve()
            if encoder_cache_dir is not None
            else None
        )
        for tag in self.OPT_ORDER:
            p = paths_by_opt.get(tag)
            if p is None:
                continue
            pp = Path(p).expanduser().resolve()
            if not pp.is_file():
                raise FileNotFoundError(f"Curriculum {tag} jsonl 不存在: {pp}")
            self._labels.append(tag)
            base_ds = BinSkelJsonlDataset(
                pp,
                max_samples=max_samples,
                require_alignment=True,
                skip_invalid=skip_invalid,
            )
            if cache_root is not None:
                cache_sub = cache_root / pp.stem
                if not (cache_sub / STAGE2_CACHE_MANIFEST_NAME).is_file():
                    raise FileNotFoundError(
                        f"Curriculum {tag}: encoder cache 不存在 "
                        f"({cache_sub / STAGE2_CACHE_MANIFEST_NAME})；"
                        f"请先运行 scripts/precompute_stage2_encoder_cache.py",
                    )
                cache = Stage2EncoderCache(
                    cache_sub, shard_lru_size=int(encoder_cache_shard_lru),
                )
                self.datasets.append(Stage2EncoderCachedDataset(base_ds, cache))
            else:
                self.datasets.append(base_ds)
        if not self.datasets:
            raise ValueError("Stage2CurriculumDataset: 至少提供一个优化档路径（建议从 O0 开始）")
        # 前缀长度累计（最多 5 档：O0/O1/O2/O3/Os，实际档数取决于传入的路径）
        self._full_cumulative: List[int] = []
        t = 0
        for d in self.datasets:
            t += len(d)
            self._full_cumulative.append(t)
        self._stage = 0

    def max_stage(self) -> int:
        return len(self.datasets) - 1

    def opt_tags_in_order(self) -> List[str]:
        """已并入的优化档标签，顺序与 :data:`OPT_ORDER` 一致（仅含实际存在的档）。"""
        return list(self._labels)

    def get_stage(self) -> int:
        return int(self._stage)

    def set_stage(self, stage: int) -> None:
        self._stage = max(0, min(int(stage), self.max_stage()))

    def active_opt_tags(self) -> List[str]:
        return list(self._labels[: self._stage + 1])

    def __len__(self) -> int:
        return int(self._full_cumulative[self._stage])

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"curriculum idx={idx} len={len(self)}")
        acc = 0
        for si, end in enumerate(self._full_cumulative[: self._stage + 1]):
            prev = acc
            acc = end
            if idx < acc:
                return self.datasets[si].__getitem__(idx - prev)
        raise RuntimeError("Stage2CurriculumDataset: 内部索引错误")


def build_stage2_train_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    skip_invalid: bool = True,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """任意 Stage2 样本 Dataset → DataLoader（含 curriculum / 单文件）。

    ``distributed=True`` 时使用 :class:`torch.utils.data.distributed.DistributedSampler`，
    此时 ``shuffle`` 由 sampler 控制（原 ``shuffle`` 传 False）。
    """
    collate_fn = (
        collate_binskel_stage2_with_skip if skip_invalid else collate_binskel_stage2_batch
    )
    sampler: Optional[Any] = None
    if distributed and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(dataset, **kw)


def make_binskel_dataloader(
    jsonl_path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    max_samples: Optional[int] = None,
    require_source: bool = False,
    skip_invalid: bool = True,
    pin_memory: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    inst_dropout_p: float = 0.0,
    edge_dropout_p: float = 0.0,
    add_func_uid: bool = True,
    cross_opt_pairs: bool = False,
) -> DataLoader:
    """构造已绑定 ``collate_fn`` 的 DataLoader。

    ``jsonl_path`` 可为**单个**路径，或**多个**路径的序列；多路径时用
    :class:`torch.utils.data.ConcatDataset` 拼成同一训练集。

    ``max_samples`` 若设置：在**每个** jsonl 文件上各自最多保留该条数有效样本
    （多文件时总上限约为 ``max_samples * 文件数``）。

    ``skip_invalid=True`` 时 ``collate_fn`` 会丢弃空样本（与
    :meth:`BinSkelJsonlDataset.__getitem__` 在运行时发现坏行时的行为一致）。
    """
    if isinstance(jsonl_path, (str, Path)):
        paths: List[Path] = [Path(jsonl_path)]
    else:
        paths = [Path(p) for p in jsonl_path]
    paths = [p.expanduser().resolve() for p in paths]

    datasets: List[Dataset] = [
        BinSkelJsonlDataset(
            p,
            max_samples=max_samples,
            require_source=require_source,
            skip_invalid=skip_invalid,
            cross_opt_pairs=cross_opt_pairs,
        )
        for p in paths
    ]
    ds: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    _kw = dict(
        inst_dropout_p=float(inst_dropout_p),
        edge_dropout_p=float(edge_dropout_p),
        add_func_uid=bool(add_func_uid),
    )
    if skip_invalid:
        collate_fn = partial(_collate_binskel_with_skip_kw, **_kw)
    else:
        collate_fn = partial(collate_binskel_batch, **_kw)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False

    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        if prefetch_factor is not None:
            kw["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kw)


def make_binskel_stage3_dataloader(
    jsonl_path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    skip_invalid: bool = True,
    pin_memory: bool = False,
    pad_token_id: int = 0,
    max_sft_seq_len: Optional[int] = None,
    fill_token_ids: Optional[torch.Tensor] = None,
    rebuild_slot_mask_when_missing: bool = True,
) -> DataLoader:
    """Stage3 SFT：jsonl 须含 ``alignment_matrix``、``sft_input_ids``、``sft_labels``、``slot_types``。

    ``fill_token_ids``：与 augment / Stage3 tokenizer 一致的 ``<|STRUCLIFT_FILL_*|>`` id，
    供 collate 精确构造 ``slot_mask``（含无 ``sft_fill_positions`` 时的骨架推断）。

    ``max_sft_seq_len``：若设置，长度超过该值的样本在索引阶段会被跳过（利于在 24GB 等显卡上控制显存）。
    """
    if isinstance(jsonl_path, (str, Path)):
        paths: List[Path] = [Path(jsonl_path)]
    else:
        paths = [Path(p) for p in jsonl_path]
    paths = [p.expanduser().resolve() for p in paths]

    datasets: List[Dataset] = [
        BinSkelJsonlDataset(
            p,
            max_samples=max_samples,
            require_sft=True,
            skip_invalid=skip_invalid,
            max_sft_seq_len=max_sft_seq_len,
        )
        for p in paths
    ]
    ds: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if len(ds) == 0:
        raise ValueError(
            "Stage3 数据集为空：jsonl 须含 alignment_matrix 与 "
            "sft_input_ids / sft_labels / slot_types（例如 FULL_STAGES_JSONL=1 生成，或 augment_binskel_sft）"
            + (
                f"；或 max_sft_seq_len={max_sft_seq_len} 过滤掉了全部样本，请调大上限或缩短 jsonl 中的序列"
                if max_sft_seq_len is not None
                else ""
            ),
        )
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]]
    if skip_invalid:
        collate_fn = partial(
            collate_binskel_stage3_with_skip,
            pad_token_id=pad_token_id,
            max_sft_seq_len=max_sft_seq_len,
            fill_token_ids=fill_token_ids,
            rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
        )
    else:
        collate_fn = partial(
            collate_binskel_stage3_batch,
            pad_token_id=pad_token_id,
            max_sft_seq_len=max_sft_seq_len,
            fill_token_ids=fill_token_ids,
            rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
        )
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)


def make_binskel_stage4_dataloader(
    jsonl_path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    skip_invalid: bool = True,
    pin_memory: bool = False,
    pad_token_id: int = 0,
    num_patterns: int = _DEFAULT_NUM_PATTERNS,
    max_sft_seq_len: Optional[int] = None,
    fill_token_ids: Optional[torch.Tensor] = None,
    rebuild_slot_mask_when_missing: bool = True,
) -> DataLoader:
    """Stage4 GRPO：与 Stage3 相同 jsonl 要求，另可选 ``bin_pattern_dist`` 等结构奖励字段。

    ``max_sft_seq_len``、``fill_token_ids``：同 :func:`make_binskel_stage3_dataloader`。
    """
    if isinstance(jsonl_path, (str, Path)):
        paths: List[Path] = [Path(jsonl_path)]
    else:
        paths = [Path(p) for p in jsonl_path]
    paths = [p.expanduser().resolve() for p in paths]

    datasets: List[Dataset] = [
        BinSkelJsonlDataset(
            p,
            max_samples=max_samples,
            require_sft=True,
            skip_invalid=skip_invalid,
            max_sft_seq_len=max_sft_seq_len,
        )
        for p in paths
    ]
    ds: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if len(ds) == 0:
        raise ValueError(
            "Stage4 数据集为空：jsonl 须含 SFT 字段（同 Stage3），"
            "并建议含 bin_pattern_dist 等结构奖励字段（FULL_STAGES_JSONL=1 或 augment --emit-rl-fields）"
            + (
                f"；或 max_sft_seq_len={max_sft_seq_len} 过滤掉了全部样本，请调大上限或缩短 jsonl 中的序列"
                if max_sft_seq_len is not None
                else ""
            ),
        )
    if skip_invalid:
        collate_fn = partial(
            collate_binskel_stage4_with_skip,
            pad_token_id=pad_token_id,
            num_patterns=num_patterns,
            max_sft_seq_len=max_sft_seq_len,
            fill_token_ids=fill_token_ids,
            rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
        )
    else:
        collate_fn = partial(
            collate_binskel_stage4_batch,
            pad_token_id=pad_token_id,
            num_patterns=num_patterns,
            max_sft_seq_len=max_sft_seq_len,
            fill_token_ids=fill_token_ids,
            rebuild_slot_mask_when_missing=rebuild_slot_mask_when_missing,
        )
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)


def make_binskel_stage2_dataloader(
    jsonl_path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    skip_invalid: bool = True,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    encoder_cache_dir: Optional[Union[str, Path]] = None,
    encoder_cache_shard_lru: int = 4,
) -> DataLoader:
    """Stage2 对齐训练用 DataLoader：样本须含 ``alignment_matrix``（及源码侧）。

    ``collate_fn`` 产出 ``M_gt``、``adj_bin``、``adj_src_gt``、``alignment_confidence``。

    ``distributed=True`` 时使用 ``DistributedSampler``（多卡 DDP 与单卡复用同接口）。

    ``encoder_cache_dir`` 非 None 时，对每个 jsonl 找同名子目录加载
    :class:`Stage2EncoderCache`，最终样本会带 ``_cache_*`` 字段供 collate 拼成
    ``cache_*`` 张量，触发 :meth:`StrucLift.encode` 的 precomputed 旁路。
    """
    if isinstance(jsonl_path, (str, Path)):
        paths: List[Path] = [Path(jsonl_path)]
    else:
        paths = [Path(p) for p in jsonl_path]
    paths = [p.expanduser().resolve() for p in paths]

    cache_root: Optional[Path] = (
        Path(encoder_cache_dir).expanduser().resolve()
        if encoder_cache_dir is not None
        else None
    )

    datasets: List[Dataset] = []
    for p in paths:
        base = BinSkelJsonlDataset(
            p,
            max_samples=max_samples,
            require_alignment=True,
            skip_invalid=skip_invalid,
        )
        if cache_root is not None:
            cache_sub = cache_root / p.stem
            if not (cache_sub / STAGE2_CACHE_MANIFEST_NAME).is_file():
                raise FileNotFoundError(
                    f"encoder cache 不存在: {cache_sub / STAGE2_CACHE_MANIFEST_NAME}"
                    f"；请先运行 scripts/precompute_stage2_encoder_cache.py",
                )
            cache = Stage2EncoderCache(
                cache_sub, shard_lru_size=int(encoder_cache_shard_lru),
            )
            datasets.append(Stage2EncoderCachedDataset(base, cache))
        else:
            datasets.append(base)
    ds: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    collate_fn = (
        collate_binskel_stage2_with_skip if skip_invalid else collate_binskel_stage2_batch
    )
    sampler: Optional[Any] = None
    if distributed and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(ds, **kw)
