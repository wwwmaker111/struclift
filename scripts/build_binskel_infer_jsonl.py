#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ELF 提取 **仅二进制侧** 的 binskel 字段，写出推理用 jsonl（无需源码、无对齐矩阵）。

与 ``build_binskel_dataset_md.py`` 的区别
--------------------------------------
- 不读 ``--src``、不要求 DWARF 行号落到 ``.c`` 文件；
- 每条样本只含 Module A/C 所需的 ``bin_*`` + 规则 ``pattern_labels``（``assign_pattern_labels``）；
- 标记 ``"infer_only": true``，供 ``infer_binskel_pipeline.py`` / ``eval_stage3_binskel.py`` 识别。

推理部署时的角色
----------------
这是「**.o/.elf → 模型可吃的张量**」那一步；不是训练监督。训练时源码仍用于
``sft_labels`` / ``M_gt``；推理只带本脚本产物即可尝试填槽反编译。

用法::

  python scripts/build_binskel_infer_jsonl.py \\
    --elf /path/to/foo.o \\
    --out /tmp/foo_infer.jsonl \\
    --func-names memcpy,strlen \\
    --opt O0

  # 或从 ELF 导出前 50 个符号函数
  python scripts/build_binskel_infer_jsonl.py --elf foo.o --out foo_infer.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elftools.elf.elffile import ELFFile

from build_binskel_dataset_md import (
    assign_pattern_labels,
    build_cfg_for_function,
    iter_functions,
)
from struclift.config import VocabConfig
from struclift.utils.graph_utils import extract_structural_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cfg_to_bin_sample(
    *,
    elf_path: Path,
    func_name: str,
    cfg: Any,
    opt_level: str,
    arch: str,
    compiler: str,
) -> Dict[str, Any]:
    """将 :class:`BinaryCFG` 打成与 ``collate_binskel_*`` 兼容的 dict。"""
    import torch

    struct_feats = extract_structural_features(cfg)
    pattern_labels = assign_pattern_labels(cfg)

    bin_opcodes: List[int] = []
    bin_ot: List[int] = []
    bin_ov: List[int] = []
    bin_bl: List[int] = []
    for bb in cfg.blocks:
        bin_bl.append(len(bb.instructions))
        for inst in bb.instructions:
            bin_opcodes.append(int(inst.opcode))
            ot = (list(inst.operand_types) + [0, 0, 0])[:3]
            vb = (list(inst.value_buckets) + [0, 0, 0])[:3]
            bin_ot.extend(int(x) for x in ot)
            bin_ov.extend(int(x) for x in vb)

    src_e = [int(e[0]) for e in cfg.edges]
    dst_e = [int(e[1]) for e in cfg.edges]
    etype = [int(e[2]) for e in cfg.edges]

    return {
        "binary": str(elf_path.resolve()),
        "func_name": func_name,
        "opt_level": opt_level,
        "arch": arch,
        "compiler": compiler,
        "infer_only": True,
        "bin_opcodes": bin_opcodes,
        "bin_operand_types": bin_ot,
        "bin_operand_values": bin_ov,
        "bin_block_lengths": bin_bl,
        "bin_edge_index": [src_e, dst_e],
        "bin_edge_type": etype,
        "bin_struct_features": struct_feats.detach().cpu().tolist(),
        "pattern_labels": [int(x) for x in pattern_labels],
    }


def _parse_func_names_arg(raw: Optional[str]) -> Optional[set]:
    if not raw:
        return None
    names = {x.strip() for x in raw.replace(",", " ").split() if x.strip()}
    return names or None


def _parse_func_names_file(path: Path) -> set:
    names: set = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0])
    return names


def iter_infer_samples(
    elf_path: Path,
    *,
    func_name_filter: Optional[set] = None,
    limit: int = 0,
    opt_level: str = "O0",
    arch: str = "x86_64",
    compiler: str = "unknown (stripped)",
    num_opcodes: int = 1024,
) -> Iterator[Dict[str, Any]]:
    """从 ELF ``.text`` 逐函数 disassemble → binskel 二进制字段。"""
    with elf_path.open("rb") as f:
        elffile = ELFFile(f)
        text_sec = elffile.get_section_by_name(".text")
        if text_sec is None:
            raise RuntimeError(f"{elf_path} 无 .text 段")
        text_bytes = text_sec.data()
        base_addr = text_sec["sh_addr"]

        n_out = 0
        for func_name, addr, size in iter_functions(elffile):
            if func_name_filter is not None and func_name not in func_name_filter:
                continue
            start = addr - base_addr
            if start < 0 or start + size > len(text_bytes):
                logger.warning("跳过 %s: 地址越界", func_name)
                continue
            code_bytes = text_bytes[start : start + size]
            cfg, _ranges = build_cfg_for_function(
                code_bytes, addr, func_name, num_opcodes=num_opcodes,
            )
            if cfg.num_blocks == 0:
                logger.warning("跳过 %s: 空 CFG", func_name)
                continue
            yield _cfg_to_bin_sample(
                elf_path=elf_path,
                func_name=func_name,
                cfg=cfg,
                opt_level=opt_level,
                arch=arch,
                compiler=compiler,
            )
            n_out += 1
            if limit > 0 and n_out >= limit:
                break


def build_infer_jsonl(
    elf_path: Path,
    out_path: Path,
    *,
    func_names: Optional[set] = None,
    limit: int = 0,
    opt_level: str = "O0",
    arch: str = "x86_64",
    compiler: str = "unknown (stripped)",
    num_opcodes: Optional[int] = None,
) -> int:
    """写出 jsonl，返回成功条数。"""
    vc = VocabConfig()
    nop = int(num_opcodes or vc.num_opcodes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for sample in iter_infer_samples(
            elf_path,
            func_name_filter=func_names,
            limit=limit,
            opt_level=opt_level,
            arch=arch,
            compiler=compiler,
            num_opcodes=nop,
        ):
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="ELF → 无源码 binskel 推理 jsonl")
    ap.add_argument("--elf", type=Path, required=True, help=".o / 可执行文件 / .so")
    ap.add_argument("--out", type=Path, required=True, help="输出 .jsonl")
    ap.add_argument(
        "--func-names",
        type=str,
        default=None,
        help="逗号或空格分隔的函数名；省略则处理 ELF 内全部 STT_FUNC",
    )
    ap.add_argument(
        "--func-names-file",
        type=Path,
        default=None,
        help="每行一个函数名",
    )
    ap.add_argument("--limit", type=int, default=0, help="最多写出 N 条（0=不限制）")
    ap.add_argument("--opt", default="O0", help="写入 opt_level 元数据")
    ap.add_argument("--arch", default="x86_64")
    ap.add_argument("--compiler", default="unknown (stripped)")
    ap.add_argument("--num-opcodes", type=int, default=None)
    args = ap.parse_args()

    elf_path = args.elf.expanduser().resolve()
    if not elf_path.is_file():
        logger.error("ELF 不存在: %s", elf_path)
        sys.exit(1)

    filt: Optional[set] = _parse_func_names_arg(args.func_names)
    if args.func_names_file is not None:
        ff = _parse_func_names_file(args.func_names_file.expanduser().resolve())
        filt = ff if filt is None else (filt & ff)

    n = build_infer_jsonl(
        elf_path,
        args.out.expanduser().resolve(),
        func_names=filt,
        limit=args.limit,
        opt_level=args.opt,
        arch=args.arch,
        compiler=args.compiler,
        num_opcodes=args.num_opcodes,
    )
    logger.info("写出 %d 条 → %s", n, args.out)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
