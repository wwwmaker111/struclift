"""
从模型生成的 C 代码提取结构度量，供 Stage4 ``R_struct`` 使用。

流程（对齐提案 §8.2.3 思路：生成码 → 可执行结构 → 与二进制侧可比）：

1. 将片段包进 ``void __struclift_gen_fn(void){ ... }``，``gcc -c`` 得到 ``.o``
2. 从 ``.o`` 中读出该符号的机器码
3. 复用仓库根目录 ``build_binskel_dataset.py`` 里的 Capstone CFG 与 ``assign_pattern_labels``
4. :func:`struclift.utils.skeleton.build_control_structure_tree` → 前序结点类型序列 + 嵌套深度
5. 由 per-BB pattern 标签构造归一化直方图 ``[C]``，与 jsonl 中的 ``bin_pattern_dist`` 维数一致

若 ``build_binskel_dataset.py`` 不可用、或编译/反汇编失败，返回 ``None``（此时调用方应将 ``r_s`` 置 0）。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from struclift.utils.skeleton import CSTNode, build_control_structure_tree

logger = logging.getLogger(__name__)

_GEN_FN = "__struclift_gen_fn"

_BOILER = """#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static void __struclift_unused(void) {{ (void)printf; (void)malloc; (void)free; }}
void {name}(void) {{
#line 1 "struclift_gen.c"
{body}
}}
"""


def _load_build_binskel() -> Optional[Any]:
    root = Path(__file__).resolve().parents[2]
    bpath = root / "build_binskel_dataset.py"
    if not bpath.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_struclift_binskel_build", bpath)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.debug("无法加载 build_binskel_dataset: %s", e)
        return None


def _extract_symbol_bytes_o(o_path: Path, sym_name: str) -> Optional[Tuple[bytes, int]]:
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        return None
    try:
        with o_path.open("rb") as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(
                ".dynsym",
            )
            if symtab is None:
                return None
            for sym in symtab.iter_symbols():
                if sym.name != sym_name:
                    continue
                addr = sym["st_value"]
                size = sym["st_size"]
                shndx = sym["st_shndx"]
                if size <= 0 or not isinstance(shndx, int) or shndx <= 0:
                    continue
                sec = elf.get_section(shndx)
                data = sec.data()
                off = int(addr)
                if 0 <= off < len(data):
                    chunk = data[off : off + int(size)]
                    if chunk:
                        return chunk, off
    except Exception as e:
        logger.debug("ELF 解析失败: %s", e)
    return None


def _cst_preorder_types(root: CSTNode) -> List[str]:
    out: List[str] = [root.type]
    for ch in root.children:
        out.extend(_cst_preorder_types(ch))
    return out


def _cst_max_depth_below_root(root: CSTNode) -> int:
    """控制结构树最大深度（根 ``function`` 深度记 0）。"""

    def walk(n: CSTNode, d: int) -> int:
        best = d
        for ch in n.children:
            best = max(best, walk(ch, d + 1))
        return best

    return walk(root, 0)


def _labels_to_dist(labels: List[int], num_classes: int) -> torch.Tensor:
    h = [0.0] * num_classes
    for lb in labels:
        li = int(lb)
        if 0 <= li < num_classes:
            h[li] += 1.0
    s = sum(h)
    if s <= 0:
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    return torch.tensor([x / s for x in h], dtype=torch.float32)


def structural_metrics_from_generated_c(
    code: str,
    *,
    num_pattern_classes: int,
    num_opcodes: int = 1024,
    gcc_timeout: int = 30,
) -> Optional[Tuple[torch.Tensor, List[str], int]]:
    """返回 ``(pattern_dist[C], cst_preorder_types, nesting_depth)`` 或 ``None``。"""
    mod = _load_build_binskel()
    if mod is None:
        return None
    build_cfg = getattr(mod, "build_cfg_for_function", None)
    assign_labels = getattr(mod, "assign_pattern_labels", None)
    if build_cfg is None or assign_labels is None:
        return None

    body = code.strip()
    if not body:
        return None

    src = _BOILER.format(name=_GEN_FN, body=body)
    c_path: Optional[Path] = None
    o_path: Optional[Path] = None
    try:
        c_f = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".c",
            delete=False,
            encoding="utf-8",
        )
        c_f.write(src)
        c_f.close()
        c_path = Path(c_f.name)
        o_fd, o_name = tempfile.mkstemp(suffix=".o")
        os.close(o_fd)
        o_path = Path(o_name)
        r = subprocess.run(
            [
                "gcc",
                "-c",
                "-O0",
                "-w",
                str(c_path),
                "-o",
                str(o_path),
            ],
            capture_output=True,
            timeout=gcc_timeout,
        )
        if r.returncode != 0:
            logger.debug(
                "gcc -c 失败: %s",
                r.stderr.decode(errors="replace")[:500],
            )
            return None

        got = _extract_symbol_bytes_o(o_path, _GEN_FN)
        if got is None:
            logger.debug("未在 .o 中找到符号 %s", _GEN_FN)
            return None
        raw, base_addr = got
        cfg, _ranges = build_cfg(raw, base_addr, _GEN_FN, num_opcodes=num_opcodes)
        n = cfg.num_blocks
        labels_list = assign_labels(cfg)
        labels_t = torch.tensor(labels_list, dtype=torch.int64)
        if n == 0:
            dist = torch.ones(num_pattern_classes, dtype=torch.float32) / num_pattern_classes
            return dist, [], 0

        src_e = [e[0] for e in cfg.edges]
        dst_e = [e[1] for e in cfg.edges]
        edge_index = torch.tensor([src_e, dst_e], dtype=torch.int64)

        cst = build_control_structure_tree(labels_t, edge_index, n)
        preorder = _cst_preorder_types(cst)
        depth = _cst_max_depth_below_root(cst)
        dist = _labels_to_dist(labels_list, num_pattern_classes)
        return dist, preorder, depth
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("编译/文件错误: %s", e)
        return None
    except Exception as e:
        logger.debug("结构提取异常: %s", e)
        return None
    finally:
        if c_path is not None:
            try:
                c_path.unlink(missing_ok=True)
            except OSError:
                pass
        if o_path is not None:
            try:
                o_path.unlink(missing_ok=True)
            except OSError:
                pass
