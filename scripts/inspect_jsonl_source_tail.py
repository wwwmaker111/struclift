#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看 jsonl **末尾**若干条样本的 ``source_file`` 形态（用于对比「前面」与「后面」是否同一套路径规则）。

默认取**最后 500 条非空行**，统计：

- 是否能用给定 ``--source-root``（可多个）按与 ``augment_binskel_sft._resolve_source_path`` 类似规则找到文件；
- 路径中锚点目录（``structlift_src_datasets`` / ``source_datasets`` 等）后的**工程名**频次。

示例::

  python3 scripts/inspect_jsonl_source_tail.py \\
    --jsonl /data/chaoni/WQM/datasets/Qwen3.6_C_train_o0.jsonl \\
    --tail 500 \\
    --source-root /data/chaoni/WQM/source_datasets
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve(
    record: Dict[str, Any], roots: List[Path],
) -> Optional[Path]:
    raw = record.get("source_file")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p.resolve()
    for source_root in roots:
        q = (source_root / p).resolve()
        if q.is_file():
            return q
        q2 = (source_root / p.name).resolve()
        if q2.is_file():
            return q2
    return None


def _project_after_anchor(sf: str, anchors: Tuple[str, ...]) -> str:
    parts = sf.replace("\\", "/").split("/")
    low = [x.lower() for x in parts]
    for a in anchors:
        al = a.lower()
        if al in low:
            i = low.index(al)
            if i + 1 < len(parts):
                return parts[i + 1]
            return "?_at_end"
    return "__NO_KNOWN_ANCHOR__"


def main() -> None:
    ap = argparse.ArgumentParser(description="检查 jsonl 末尾样本的 source_file")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument(
        "--tail",
        type=int,
        default=500,
        metavar="N",
        help="取最后 N 条非空 jsonl（默认 500）",
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        action="append",
        dest="source_roots",
        help="可多传，与 augment 一致（默认不包含，仅做路径归类）",
    )
    args = ap.parse_args()

    path: Path = args.jsonl.expanduser().resolve()
    n_keep = max(1, int(args.tail))
    roots = [Path(p).expanduser().resolve() for p in (args.source_roots or [])]
    anchors = (
        "structlift_src_datasets",
        "source_datasets",
        "datasets",
    )

    buf: deque[str] = deque(maxlen=n_keep)
    total_nonempty = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total_nonempty += 1
            buf.append(s)

    print(f"文件: {path}")
    print(f"非空行总数: {total_nonempty}")
    print(f"分析窗口: **最后 {len(buf)}** 条非空行\n")

    proj_c = Counter()
    resolve_ok = 0
    no_sf = 0
    prefix_sample: Dict[str, str] = {}

    anchors_found = Counter()

    for s in buf:
        d = json.loads(s)
        sf = str(d.get("source_file") or "").strip()
        if not sf:
            no_sf += 1
            continue
        rp = _resolve(d, roots) if roots else None
        if rp is not None:
            resolve_ok += 1
        pj = _project_after_anchor(sf, anchors)
        proj_c[pj] += 1
        low_parts = [x.lower() for x in sf.replace("\\", "/").split("/")]
        for a in anchors:
            if a.lower() in low_parts:
                anchors_found[a] += 1
                break

        pr = sf.replace("\\", "/")[:180]
        if pr not in prefix_sample:
            prefix_sample[pr] = pj

    print("— 末尾窗口内路径锚点命中（每条样本最多计一个已知锚点名）—")
    if anchors_found:
        for k, v in anchors_found.most_common():
            print(f"  含 `{k}`: {v}")
    else:
        print("  （未命中预设锚点 structlift_src_datasets/source_datasets）")

    print("\n— 锚点后再取一段「工程目录」频次（用于看是否不止 ffmpeg）—")
    for name, cnt in proj_c.most_common(40):
        print(f"  {name!s:40s} {cnt}")

    if roots:
        print(f"\n— 给定 {len(roots)} 个 --source-root 时可解析的文件数: {resolve_ok}/{len(buf)-no_sf}")

    print(f"\n无 source_file 字段或为空（在窗口内）: {no_sf}")
    print("\n— 末尾窗口内不重样的路径前缀示例（至多 25 条）—")
    for i, (pr, pj) in enumerate(prefix_sample.items()):
        if i >= 25:
            break
        print(f"  [{pj}] {pr}")

    print(
        "\n说明: 若 `__NO_KNOWN_ANCHOR__` 很多但路径仍像 `/mnt/e/structlift_src_datasets/...`,"
        " 说明锚点名已是 structlift_src_datasets；工程名多在上一段频次里。"
    )


if __name__ == "__main__":
    main()
