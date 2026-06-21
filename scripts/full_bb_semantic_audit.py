#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 audit_bundle .md 中**每一个 BB** 打语义审计标签（全量 649 可一次导出）。

说明：
- **无法**在不用模型/形式化工具的情况下给出「汇编 ≡ C 语句」的数学真值；
- 本脚本输出：**规则分层**（与 audit_bundle_bb_heuristic_check 一致）+ 可选 **Ollama 三分类**（MATCH/MISMATCH/UNCLEAR）。

用法:
  python3 scripts/full_bb_semantic_audit.py audit_bundle_35_full.md -o semantic_bb_649.jsonl
  python3 scripts/full_bb_semantic_audit.py audit_bundle_35_full.md -o out.jsonl --ollama qwen2.5:7b --limit 20

WSL 若报 PermissionError: 'ollama'：Linux 下 ollama 不可执行或未安装，可改用 HTTP（本机已跑 ollama 服务时）:
  python3 scripts/full_bb_semantic_audit.py ... -o out.jsonl --ollama qwen2.5:7b \\
    --ollama-url http://127.0.0.1:11434 --limit 20

或指定可执行文件: --ollama-bin /usr/local/bin/ollama  或  export OLLAMA_BIN=...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_bundle_bb_heuristic_check as abc  # noqa: E402


OLLAMA_PROMPT = """你是编译器与逆向助手。只根据下面「C 片段」与「x86 汇编（| 分隔）」判断：
汇编是否**主要**在实现该 C 片段所描述的那条语句的语义（允许序言/收尾少量无关指令）。

只输出一行，且必须是以下之一（不要其它文字）：
MATCH
MISMATCH
UNCLEAR

[C]
{c}

[ASM]
{asm}
"""


def _classify_one(r: dict) -> dict:
    if r["unaligned"]:
        return {
            "semantic_tier": "UNALIGNED",
            "hi_rule": None,
            "hi_msg": None,
            "coarse_rule": None,
            "coarse_msg": None,
        }
    if not r["asm"].strip():
        return {
            "semantic_tier": "ALIGNED_NO_ASM",
            "hi_rule": None,
            "hi_msg": None,
            "coarse_rule": None,
            "coarse_msg": None,
        }
    hi_msg = None
    hi_name = None
    for name, fn in abc.HI_RULES:
        m = fn(r["c_text"], r["asm"])
        if m:
            hi_name, hi_msg = name, m
            break
    if hi_msg:
        return {
            "semantic_tier": "HI_CONFLICT",
            "hi_rule": hi_name,
            "hi_msg": hi_msg,
            "coarse_rule": None,
            "coarse_msg": None,
        }
    for name, fn in abc.COARSE_RULES:
        m = fn(r["c_text"], r["asm"])
        if m:
            return {
                "semantic_tier": "COARSE_ALIGN",
                "hi_rule": None,
                "hi_msg": None,
                "coarse_rule": name,
                "coarse_msg": m,
            }
    return {
        "semantic_tier": "RULES_CLEAN",
        "hi_rule": None,
        "hi_msg": None,
        "coarse_rule": None,
        "coarse_msg": None,
    }


def _parse_llm_line(text: str) -> str:
    out = text.strip().splitlines()
    if not out:
        return "ERROR:empty_response"
    line = out[-1].strip().upper()
    for token in ("MATCH", "MISMATCH", "UNCLEAR"):
        if token in line:
            return token
    return f"UNCLEAR_RAW:{line[:80]}"


def _ollama_classify_http(
    base_url: str, model: str, c_text: str, asm: str, timeout: int
) -> str:
    prompt = OLLAMA_PROMPT.format(
        c=c_text[:4000] if c_text else "(空)",
        asm=asm[:8000] if asm else "(空)",
    )
    url = base_url.rstrip("/") + "/api/generate"
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return _parse_llm_line(data.get("response") or "")
    except urllib.error.URLError as e:
        return f"ERROR:http:{e.reason!s}"[:120]
    except json.JSONDecodeError as e:
        return f"ERROR:json:{e!s}"[:120]


def _ollama_classify_cli(
    ollama_bin: str, model: str, c_text: str, asm: str, timeout: int
) -> str:
    prompt = OLLAMA_PROMPT.format(
        c=c_text[:4000] if c_text else "(空)",
        asm=asm[:8000] if asm else "(空)",
    )
    try:
        p = subprocess.run(
            [ollama_bin, "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if p.returncode != 0 and not (p.stdout or "").strip():
            err = (p.stderr or "")[:200]
            return f"ERROR:rc{p.returncode}:{err}"
        return _parse_llm_line(p.stdout or "")
    except subprocess.TimeoutExpired:
        return "ERROR:timeout"
    except FileNotFoundError:
        return "ERROR:ollama_not_found"
    except PermissionError:
        return "ERROR:permission_denied_cli"


def _resolve_ollama_executable(bin_arg: str) -> str | None:
    """返回可执行路径，否则 None。"""
    if os.path.isfile(bin_arg) and os.access(bin_arg, os.X_OK):
        return os.path.abspath(bin_arg)
    w = shutil.which(bin_arg)
    if w and os.access(w, os.X_OK):
        return w
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="全量 BB 语义审计 JSONL")
    ap.add_argument("md", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent / "audit_bundle_35_full.md")
    ap.add_argument("-o", "--output", type=Path, required=True, help="输出 .jsonl")
    ap.add_argument("--ollama", type=str, default="", metavar="MODEL", help="若设置则对每条调用模型（很慢）")
    ap.add_argument(
        "--ollama-url",
        type=str,
        default="",
        metavar="BASE",
        help="用 HTTP 调 Ollama，例如 http://127.0.0.1:11434（避免 WSL 下 ollama CLI 无执行权限）",
    )
    ap.add_argument(
        "--ollama-bin",
        type=str,
        default="",
        metavar="PATH",
        help="ollama 可执行文件路径；默认环境变量 OLLAMA_BIN 或命令名 ollama",
    )
    ap.add_argument("--ollama-timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 BB（0=全部）")
    args = ap.parse_args()

    if not args.md.is_file():
        print(f"找不到: {args.md}", file=sys.stderr)
        return 1

    ollama_cli: str | None = None
    if args.ollama and not args.ollama_url.strip():
        name = args.ollama_bin.strip() or os.environ.get("OLLAMA_BIN", "").strip() or "ollama"
        ollama_cli = _resolve_ollama_executable(name)
        if not ollama_cli:
            print(
                "错误: 找不到可执行的 ollama CLI。\n"
                f"  已尝试: {name!r}\n"
                "  解决: (1) chmod +x 并安装 Linux 版 ollama；或\n"
                "       (2) 使用 --ollama-url http://127.0.0.1:11434（本机 Ollama 服务已监听时）",
                file=sys.stderr,
            )
            return 1

    rows = abc.parse_bundle(args.md)
    if args.limit > 0:
        rows = rows[: args.limit]

    counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            rec = {
                "sample_key": f"{r['jsonl']}#line_idx={r['line_idx']}",
                "func_name": r["func"],
                "bb": r["bb"],
                "stmt_meta": r["stmt_meta"],
                "dwarf_lines": r["dwarf"],
                "c_text": r["c_text"],
                "asm": r["asm"],
            }
            cls = _classify_one(r)
            rec.update(cls)
            rec["llm_verdict"] = None
            if args.ollama and not r["unaligned"] and r["asm"].strip():
                if args.ollama_url.strip():
                    rec["llm_verdict"] = _ollama_classify_http(
                        args.ollama_url.strip(),
                        args.ollama,
                        r["c_text"],
                        r["asm"],
                        args.ollama_timeout,
                    )
                elif ollama_cli:
                    rec["llm_verdict"] = _ollama_classify_cli(
                        ollama_cli,
                        args.ollama,
                        r["c_text"],
                        r["asm"],
                        args.ollama_timeout,
                    )
            tier = rec["semantic_tier"]
            counts[tier] = counts.get(tier, 0) + 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  已写 {i + 1}/{len(rows)}", file=sys.stderr)

    print(f"写出: {args.output}  共 {len(rows)} 条")
    print("分层统计:")
    for k in sorted(counts.keys(), key=lambda x: (-counts[x], x)):
        print(f"  {k}: {counts[k]}")
    if args.ollama:
        mode = f"HTTP {args.ollama_url}" if args.ollama_url.strip() else f"CLI {ollama_cli}"
        print(f"(已含 llm_verdict；{mode}；请人工 spot-check MISMATCH)")
    else:
        print("未跑 --ollama；RULES_CLEAN 仍不等于语义已证实正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
