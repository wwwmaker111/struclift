# 服务器 WQM 训练环境（node4 等）

记录目的：避免在登录节点上使用 **系统 `python3`**（无 `torch_geometric` 等依赖），与已解决的 `ModuleNotFoundError: torch_geometric` 一致。

## 推荐 Python 解释器（无需先 conda activate）

```text
/data/chaoni/miniconda3/envs/wqm_struc/bin/python3
```

## Conda 环境名与路径

- 环境名：`wqm_struc`
- 根目录：`/data/chaoni/miniconda3`（与 `~/miniconda3` 不是同一套时，**必须**用绝对路径或上面解释器。）

先激活再跑时：

```bash
source /data/chaoni/miniconda3/etc/profile.d/conda.sh
conda activate wqm_struc
```

## 工程根目录

```text
/data/chaoni/WQM/model_code/structlift
```

## 测时脚本 profile_stage2_batchsize_256x3（完整一条）

在服务器上直接复制执行（已含正确解释器 + GPU）：

```bash
cd /data/chaoni/WQM/model_code/structlift && \
CUDA_VISIBLE_DEVICES=0 /data/chaoni/miniconda3/envs/wqm_struc/bin/python3 scripts/profile_stage2_batchsize_256x3.py \
  --jsonl-o0 /data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl \
  --init-from /data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt \
  --max-samples 256 \
  --batch-sizes 8 16 32 \
  --epochs 3 \
  --num-workers 4 \
  --prefetch-factor 4
```

## Stage2 多卡 DDP（`WORLD_SIZE>1` 自动启用）

- **`--batch-size` 为每卡（per-GPU）**；等效总 batch = `batch_size × 卡数`（若与单卡同训可比，可改为半 batch 以接近总 batch 不变，视实验而定）。
- 使用 **`torchrun`**，并用 **`CUDA_VISIBLE_DEVICES`** 选物理卡（如 0 与 7 映射为进程内 `cuda:0` 与 `cuda:1`）：

```bash
cd /data/chaoni/WQM/model_code/structlift && \
CUDA_VISIBLE_DEVICES=0,7 /data/chaoni/miniconda3/envs/wqm_struc/bin/torchrun \
  --standalone --nproc_per_node=2 \
  scripts/train_stage2_binskel.py \
  --curriculum \
  --jsonl-o0 /data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl \
  --jsonl-o1 /data/chaoni/WQM/datasets/AB_train_o1_2048.jsonl \
  --jsonl-o2 /data/chaoni/WQM/datasets/AB_train_o2_2048.jsonl \
  --jsonl-o3 /data/chaoni/WQM/datasets/AB_train_o3_2048.jsonl \
  --jsonl-os /data/chaoni/WQM/datasets/AB_train_os_2048.jsonl \
  --curriculum-epochs-per-stage 8 \
  --val-jsonl /data/chaoni/WQM/datasets/AB_val.jsonl \
  --init-from /data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt \
  --save-dir /data/chaoni/WQM/checkpoints/stage2_ddp2 \
  --device cuda --batch-size 64 --epochs 40 --num-workers 4 --prefetch-factor 4
```

- **验证集** 仅在 **rank0** 上跑、**checkpoint** 仅 **rank0** 写；**NCCL 超时** 默认 4h（`PG_TIMEOUT_SECONDS` 可调，同 Stage1）。

## 易错点

- 在 shell 里只打 `python3` 且 **未** `conda activate`：会落到 **/usr** 等系统 Python，会缺 PyG。
