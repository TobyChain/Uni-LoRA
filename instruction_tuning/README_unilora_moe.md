# Qwen MoE with Uni-LoRA Training Pipeline

本实现将 Uni-LoRA 架构集成到 Qwen MoE 模型中，实现参数高效的微调。

## 核心思想

Uni-LoRA 使用**共享向量库**（Shared Vector Bank）和**专家特定投影**（Expert-Specific Projections）来减少内存占用，同时保持专家特异性。

### 数学公式

对于 MoE 专家 $e$ 中的标准 Linear 层，权重更新 $\Delta W_e$ 为：

$$\Delta W_e = s \cdot (P_e \cdot v^T)$$

其中：
- $v \in \mathbb{R}^{1 \times r}$ 是**共享向量库**（Rank-1 或低秩），在所有专家之间共享
- $P_e \in \mathbb{R}^{d_{out} \times r}$ 是**投影矩阵**，每个专家 $e$ 独有
- $s = \alpha / r$ 是缩放因子

## 文件说明

### 1. `modeling_unilora_moe.py`

核心实现文件，包含：

- **`UniLoRALinear`**: 替换 MoE 专家中 Linear 层的 Uni-LoRA 线性层
- **`UniLoRAMoEExpert`**: 包装单个 MoE 专家，将 `gate_proj`, `up_proj`, `down_proj` 替换为 Uni-LoRA 版本
- **`UniLoRAMoELayer`**: 包装 Qwen MoE SparseMoeBlock，应用 Uni-LoRA 到所有专家
- **`apply_unilora_to_qwen_moe()`**: 遍历模型并替换 MoE 层的函数
- **`freeze_router()`**: 冻结/解冻路由器（gate）
- **`freeze_unilora_adapters()`**: 冻结/解冻 Uni-LoRA 适配器参数

### 2. `train_qwen_moe_unilora.py`

完整的训练脚本，支持：

- 4-bit 量化（NF4）加载模型
- 应用 Uni-LoRA 到 MoE 层
- 两阶段训练：
  - **阶段 1**: 冻结路由器，仅训练 $v$ 和 $P_e$
  - **阶段 2**: 冻结 $v$ 和 $P_e$，解冻路由器以学习更好的调度
- 与 `accelerate` 和 `deepspeed` 兼容

### 3. `ds_config.json` 和 `ds_config_zero3.json`

DeepSpeed 配置文件：

- **`ds_config.json`**: ZeRO-2 配置，适合 24GB VRAM
- **`ds_config_zero3.json`**: ZeRO-3 配置，带 CPU offload，适合更大模型或更少内存

### 4. `run_qwen_moe_unilora.sh`

启动训练脚本，包含：

- 可配置的超参数
- 两阶段训练流程
- 自动检查点管理

## 使用方法

### 环境要求

```bash
pip install torch transformers accelerate deepspeed bitsandbytes datasets pandas
```

### 快速开始

#### 阶段 1: 训练 Uni-LoRA 适配器

```bash
# 使用 ZeRO-2（推荐用于 4x RTX 3090）
bash run_qwen_moe_unilora.sh

# 或手动运行
deepspeed --num_gpus=4 train_qwen_moe_unilora.py \
    --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --dataset alpaca \
    --output_dir ./output/stage1 \
    --deepspeed ds_config.json \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --rank 64 \
    --alpha 16.0 \
    --training_stage 1 \
    --gradient_checkpointing
```

#### 阶段 2: 训练路由器

修改 `run_qwen_moe_unilora.sh` 中的 `TRAINING_STAGE=2`，然后运行：

```bash
bash run_qwen_moe_unilora.sh
```

### 自定义配置

编辑 `run_qwen_moe_unilora.sh` 中的变量：

```bash
MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B-Chat"  # 模型路径
OUTPUT_DIR="./output/qwen_moe_unilora"     # 输出目录
DATASET="alpaca"                           # 数据集
RANK=64                                    # Uni-LoRA rank
ALPHA=16.0                                 # 缩放因子
TRAINING_STAGE=1                           # 训练阶段 (1 或 2)
DS_CONFIG="ds_config.json"                 # DeepSpeed 配置
```

### 使用 ZeRO-3（更多内存优化）

如果遇到内存不足，使用 ZeRO-3 配置：

```bash
# 修改 run_qwen_moe_unilora.sh 中的 DS_CONFIG
DS_CONFIG="ds_config_zero3.json"

# 然后运行
bash run_qwen_moe_unilora.sh
```

## 两阶段训练详解

### 阶段 1: 训练适配器（冻结路由器）

**目标**: 学习共享向量 $v$ 和专家特定投影 $P_e$

**冻结的参数**:
- 所有基础模型参数
- MoE 路由器（gate）

**训练的参数**:
- 共享向量库 $v$
- 每个专家的投影矩阵 $P_e$

**为什么这样做**:
- 首先学习如何通过共享向量和专家特定投影来适应任务
- 路由器保持预训练状态，确保专家选择合理

### 阶段 2: 训练路由器（冻结适配器）

**目标**: 学习更好的专家调度策略

**冻结的参数**:
- 所有基础模型参数
- 共享向量库 $v$
- 专家特定投影 $P_e$

**训练的参数**:
- MoE 路由器（gate）

**为什么这样做**:
- 在适配器固定后，优化路由器以更好地利用已学习的适配器
- 允许模型学习针对特定任务的专家调度策略

## 内存优化

### 4-bit 量化

使用 `BitsAndBytesConfig` 进行 4-bit NF4 量化：

```python
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

### DeepSpeed ZeRO

- **ZeRO-2**: 分片优化器和梯度，适合大多数情况
- **ZeRO-3**: 额外分片参数，适合更大模型或更少内存

### 梯度检查点

启用 `--gradient_checkpointing` 以牺牲计算时间换取内存节省。

## 检查点管理

训练脚本会自动：

- 定期保存检查点（`--save_steps`）
- 保存最佳模型（`--load_best_model_at_end`）
- 限制保存的检查点数量（`--save_total_limit`）

阶段 2 会自动从阶段 1 的最佳检查点加载。

## 故障排除

### 内存不足

1. 减少 `per_device_train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 使用 `ds_config_zero3.json`（ZeRO-3）
4. 减少 `source_max_len` 和 `target_max_len`

### 找不到 MoE 层

确保模型名称包含 "moe" 或 "sparse_moe"。如果使用不同的模型结构，可能需要调整 `apply_unilora_to_qwen_moe()` 中的层查找逻辑。

### 训练不稳定

1. 降低学习率
2. 增加 warmup 步数
3. 使用梯度裁剪（`--max_grad_norm`）

## 参数说明

### Uni-LoRA 参数

- `--rank`: 低秩适应的秩（默认: 64）
- `--alpha`: 缩放因子（默认: 16.0）
- `--training_stage`: 训练阶段，1 或 2（默认: 1）

### 训练参数

- `--per_device_train_batch_size`: 每个 GPU 的批次大小（默认: 1）
- `--gradient_accumulation_steps`: 梯度累积步数（默认: 16）
- `--learning_rate`: 学习率（默认: 2e-4）
- `--num_train_epochs`: 训练轮数（默认: 3）

## 输出文件

训练完成后，输出目录包含：

- `checkpoint-*/`: 训练检查点
- `unilora_params.pt`: Uni-LoRA 特定参数（共享向量等）
- `training_args.bin`: 训练参数
- `trainer_state.json`: 训练状态
- `*.log`: 训练日志

## 引用

如果使用本实现，请引用原始 Uni-LoRA 论文和 Qwen MoE 模型。

## 许可证

请遵循原始 Uni-LoRA 和 Qwen 模型的许可证要求。

