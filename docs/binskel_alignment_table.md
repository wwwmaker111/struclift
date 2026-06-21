# BinSkel 对齐效果汇总表（与终端 `report_alignment_all_datasets.py` 同列）

目录：`$OUT`（默认 `~/structlift_datasets`）。

**说明**：下表 **curl** 行为 2025-03 快照（ELF 含 DWARF、`compile_curl.sh` + `build_curl_binskel.sh` 全量对齐）；**含 M_gt%** 与 `report_alignment_all_datasets.py` / `validate_alignment.py` 第 5 项一致，**curl 四档见表末脚注补数命令**。其它项目仍以本机 `report_alignment_all_datasets.py` 输出为准可刷新。

| 数据集 | 样本 | BB覆盖率% | 平均置信 | 高置信% | 低置信% | 顺序% | 控制流% | 含M_gt% |
|--------|------|-----------|----------|---------|---------|-------|---------|---------|
| **zlib** | | | | | | | | |
| binskel_zlib_o0.jsonl | 149 | 96.92 | 0.929 | 92.59 | 3.34 | 93.83 | 87.58 | 100.0 |
| binskel_zlib_o1.jsonl | 149 | 96.92 | 0.688 | 52.14 | 0.67 | 93.83 | 87.58 | 99.3 |
| binskel_zlib_o2.jsonl | 149 | 96.92 | 0.688 | 52.14 | 0.67 | 93.83 | 87.58 | 99.3 |
| binskel_zlib_o3.jsonl | 149 | 96.92 | 0.688 | 52.14 | 0.67 | 93.83 | 87.58 | 99.3 |
| **sqlite** | | | | | | | | |
| binskel_sqlite_o0.jsonl | 2869 | 98.54 | 0.922 | 92.46 | 2.90 | 95.02 | 86.44 | 100.0 |
| binskel_sqlite_o1.jsonl | 2850 | 96.02 | 0.685 | 51.90 | 0.00 | 95.29 | 88.83 | 100.0 |
| binskel_sqlite_o2.jsonl | 2850 | 96.47 | 0.685 | 52.16 | 0.00 | 95.55 | 88.98 | 100.0 |
| binskel_sqlite_o3.jsonl | 2804 | 96.04 | 0.888 | 88.99 | 3.29 | 95.40 | 88.95 | 100.0 |
| **busybox** | | | | | | | | |
| binskel_busybox_o0.jsonl | 4277 | 94.97 | 0.820 | 78.01 | 4.80 | 95.45 | 91.82 | 99.0 |
| binskel_busybox_o1.jsonl | 4274 | 95.27 | 0.653 | 46.42 | 3.69 | 95.45 | 92.12 | 98.3 |
| binskel_busybox_o2.jsonl | 4277 | 95.39 | 0.654 | 46.41 | 3.70 | 95.59 | 92.16 | 98.5 |
| binskel_busybox_o3.jsonl | 4277 | 95.43 | 0.653 | 45.92 | 3.82 | 95.62 | 92.37 | 98.2 |
| **openssl** | | | | | | | | |
| binskel_openssl_o0.jsonl | 16389 | 96.38 | 0.889 | 87.99 | 3.35 | 93.28 | 90.73 | 94.0 |
| binskel_openssl_o1.jsonl | 15976 | 96.54 | 0.692 | 67.43 | 3.29 | 95.93 | 91.11 | 95.7 |
| binskel_openssl_o2.jsonl | 15979 | 95.26 | 0.882 | 87.03 | 4.24 | 95.85 | 91.15 | 95.9 |
| binskel_openssl_o3.jsonl | 15970 | 95.35 | 0.880 | 86.74 | 4.27 | 95.91 | 91.25 | 96.0 |
| **coreutils** | | | | | | | | |
| binskel_coreutils_o0.jsonl | 2552 | 95.96 | 0.691 | 55.55 | 9.25 | 95.97 | 86.13 | 95.7 |
| binskel_coreutils_o1.jsonl | 2506 | 93.99 | 0.681 | 57.02 | 10.64 | 96.11 | 88.40 | 96.5 |
| binskel_coreutils_o2.jsonl | 2508 | 94.07 | 0.680 | 57.22 | 10.72 | 96.07 | 88.39 | 96.6 |
| binskel_coreutils_o3.jsonl | 2515 | 94.01 | 0.680 | 57.00 | 10.63 | 96.07 | 88.74 | 96.7 |
| **curl** | | | | | | | | |
| binskel_curl_o0.jsonl | 1691 | 84.10 | 0.640 | 61.00 | 1.40 | 93.50 | 82.20 | 见注① |
| binskel_curl_o1.jsonl | 1691 | 83.30 | 0.630 | 56.90 | 1.40 | 93.70 | 85.90 | 见注① |
| binskel_curl_o2.jsonl | 1691 | 83.40 | 0.630 | 55.70 | 1.40 | 93.60 | 86.50 | 见注① |
| binskel_curl_o3.jsonl | 1691 | 83.60 | 0.630 | 55.80 | 1.40 | 93.50 | 86.40 | 见注① |

**注①（含 M_gt%）**：上表 curl 前 8 列来自 `validate_alignment.py` 与 `report` 相同算法；**第 9 列**请在本机执行下列任一方式填入（或直接把数字替换「见注①」）：

```bash
# 方式 A：汇总表 CSV（第 10 列为 has_matrix_pct）
python3 scripts/report_alignment_all_datasets.py ~/structlift_datasets --csv | grep binskel_curl

# 方式 B：每档第 5 行「含 M_gt …%」（validate_alignment 已打印）
for o in 0 1 2 3; do python3 scripts/validate_alignment.py ~/structlift_datasets/binskel_curl_o${o}.jsonl | grep '含 M_gt'; done
```

（若你本地 `wc -l` 与 **1691** 不一致，以本机样本数为准更新第 2 列。）

| **libxml2**（可选） | | | | | | | | |
| binskel_libxml2_o0.jsonl | … | … | … | … | … | … | … | … |

（若已生成，同样由 `report_alignment_all_datasets.py` 输出。）

---

## 高置信样本人工抽审（除 libxml2 外六项目）

每项目默认抽 **3** 条（`validate_sample` 下 mean_conf / high_W_ratio / BB 覆盖率分层筛选），导出源码 + 反汇编 + `M_gt`/`W` 到**一个文本文件**：

```bash
cd /mnt/e/structlift
python3 scripts/export_high_confidence_audit.py ~/structlift_datasets -o ~/structlift_datasets/high_confidence_audit.txt
# 默认 binskel_*_o0.jsonl；改档位: --opt 1
```

输出路径可自定；缺 jsonl 的项目会在报告中标注跳过。
