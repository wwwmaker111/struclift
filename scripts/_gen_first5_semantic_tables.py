"""一次性：从 manual_bb_audit 前 5 槽（zlib_0…zlib_os）写语义对齐 md 到 stdout。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from check_alignment_semantics import resolve_binary_path  # noqa: E402
from export_bb_semantic_alignment_table_md import build_table_md  # noqa: E402


def load_line(jp: Path, target_idx: int) -> dict | None:
    idx = -1
    with jp.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            idx += 1
            if idx == target_idx:
                return json.loads(line)
    return None


def main() -> int:
    root = Path(r"E:/structlift_datasets")
    src = REPO_ROOT / "zlib_build"
    bin_root = root
    jobs = [
        ("zlib_0", "binskel_zlib_o0.jsonl", 1),
        ("zlib_1", "binskel_zlib_o1.jsonl", 0),
        ("zlib_2", "binskel_zlib_o2.jsonl", 0),
        ("zlib_3", "binskel_zlib_o3.jsonl", 0),
        ("zlib_os", "binskel_zlib_os.jsonl", 0),
    ]
    chunks = [
        "# 前 5 个 DATASET（zlib_0 … zlib_os）语义对齐表\n",
        "\n",
        "来源：`E:/structlift_datasets/manual_bb_audit_45x1_full.txt` 中各槽 path + line_idx。\n",
        "\n---\n\n",
    ]
    for label, name, li in jobs:
        jp = root / name
        if not jp.is_file():
            chunks.append(f"## {label}\n\n*跳过：找不到 `{jp}`*\n\n---\n\n")
            continue
        d = load_line(jp, li)
        if not d:
            chunks.append(f"## {label}\n\n*跳过：line_idx={li} 无数据*\n\n---\n\n")
            continue
        d["_jsonl_path"] = str(jp)
        elf = resolve_binary_path(d, bin_root)
        md = build_table_md(
            d,
            src if src.is_dir() else None,
            elf,
            bin_root,
            all_bb=True,
            num_bb=13,
            auto_verdict=True,
            simple_verdict=False,
            dataset_banner=None,
            title_suffix=f"与你先前 {label} 一致",
        )
        chunks.append(md.rstrip() + "\n\n---\n\n")
    sys.stdout.write("".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
