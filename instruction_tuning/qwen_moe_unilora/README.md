# Qwen MoE with Uni-LoRA

将 Uni-LoRA 架构集成到 Qwen MoE 模型中的完整实现，支持参数高效的微调。

## 📁 目录结构

```
qwen_moe_unilora/
├── modeling/              # 模型实现
│   └── modeling_unilora_moe.py    # Uni-LoRA MoE 核心实现
├── training/              # 训练相关
│   ├── train_unilora_moe.py       # 训练脚本
│   └── finetune_qwem1.5_moe_A2.7_unilora_1229.sh  # 训练启动脚本
├── configs/               # 配置文件
│   ├── ds_config.json             # DeepSpeed ZeRO-2 配置
│   └── ds_config_zero3.json       # DeepSpeed ZeRO-3 配置
├── testing/               # 测试相关
│   ├── test_model_with_fastchat.py # FastChat 测试脚本
│   ├── test_model.sh              # 测试启动脚本
│   └── test_prompts.json          # 测试提示集合
└── docs/                  # 文档
    ├── README_unilora_moe.md      # 主要文档
    └── README_TESTING.md          # 测试文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
pip install torch transformers accelerate deepspeed bitsandbytes datasets pandas fschat
```

### 2. 训练模型

```bash
cd training
bash finetune_qwem1.5_moe_A2.7_unilora_1229.sh
```

### 3. 测试模型

```bash
cd testing
bash test_model.sh
```

## 📖 详细文档

- **主要文档**: [docs/README_unilora_moe.md](docs/README_unilora_moe.md)
- **测试文档**: [docs/README_TESTING.md](docs/README_TESTING.md)

## 🔧 核心特性

### Uni-LoRA 架构

- **共享向量库**: 所有专家共享一个低秩向量 `v`
- **专家特定投影**: 每个专家有独特的投影矩阵 `P_e`
- **数学公式**: $\Delta W_e = s \cdot (P_e \cdot v^T)$

### 两阶段训练

1. **阶段 1**: 冻结路由器，训练适配器（`v` 和 `P_e`）
2. **阶段 2**: 冻结适配器，训练路由器

### 内存优化

- 4-bit 量化 (NF4)
- DeepSpeed ZeRO-2/3
- 梯度检查点

## 📝 使用示例

### 训练

```bash
cd training
deepspeed --num_gpus=4 train_unilora_moe.py \
    --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --dataset alpaca-clean \
    --output_dir ./output/stage1 \
    --deepspeed ../configs/ds_config.json \
    --rank 64 \
    --alpha 16.0 \
    --training_stage 1
```

### 测试

```bash
cd testing
python test_model_with_fastchat.py \
    --base-model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --trained-model ../output/stage1/checkpoint-XXX \
    --test-prompts-file test_prompts.json \
    --output-dir ../test_results
```

## 📊 数据集

默认使用 `alpaca-clean` 数据集，支持：
- `alpaca` / `alpaca-clean`
- `chip2`
- `self-instruct`
- `hh-rlhf`
- `oasst1`
- 本地 JSON/CSV 文件

## 🔍 文件说明

### modeling/
- `modeling_unilora_moe.py`: 实现 UniLoRALinear, UniLoRAMoEExpert, UniLoRAMoELayer

### training/
- `train_unilora_moe.py`: 完整的训练脚本，支持两阶段训练
- `finetune_qwem1.5_moe_A2.7_unilora_1229.sh`: 训练启动脚本

### configs/
- `ds_config.json`: DeepSpeed ZeRO-2 配置（推荐用于 4x RTX 3090）
- `ds_config_zero3.json`: DeepSpeed ZeRO-3 配置（带 CPU offload）

### testing/
- `test_model_with_fastchat.py`: 使用 FastChat 进行模型测试和对比
- `test_model.sh`: 一键测试脚本
- `test_prompts.json`: 预定义的测试提示集合

## ⚙️ 配置说明

### 训练参数

- `--rank`: Uni-LoRA 秩（默认: 64）
- `--alpha`: 缩放因子（默认: 16.0）
- `--training_stage`: 训练阶段（1 或 2）

### DeepSpeed 配置

- **ZeRO-2**: 适合大多数情况，内存占用适中
- **ZeRO-3**: 适合更大模型或更少内存，支持 CPU offload

## 📈 输出文件

训练完成后：
- `output/stage1/`: 阶段 1 检查点
- `output/stage2/`: 阶段 2 检查点
- `unilora_params.pt`: Uni-LoRA 特定参数

测试完成后：
- `test_results/base_model_results.json`: 原始模型响应
- `test_results/trained_model_results.json`: 训练后模型响应
- `test_results/comparison.json`: 对比结果

## 🐛 故障排除

### 导入错误
确保从正确的目录运行脚本，或添加路径：
```python
import sys
sys.path.insert(0, 'path/to/qwen_moe_unilora')
```

### 内存不足
- 使用 ZeRO-3 配置
- 减少批次大小
- 增加梯度累积步数

### 模型加载失败
- 检查模型路径
- 确保有足够的 GPU 内存
- 检查 4-bit 量化配置

## 📄 许可证

请遵循原始 Uni-LoRA 和 Qwen 模型的许可证要求。

## 🙏 致谢

- [Uni-LoRA](https://github.com/KaiyangLi1992/Uni-LoRA)
- [Qwen](https://github.com/QwenLM/Qwen)
- [FastChat](https://github.com/lm-sys/FastChat)

