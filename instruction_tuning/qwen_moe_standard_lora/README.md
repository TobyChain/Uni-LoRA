# Standard LoRA for Qwen MoE

使用 HuggingFace PEFT 库的标准 LoRA 来微调 Qwen1.5-MoE-A2.7B-Chat 模型。

## 优化方法

本项目集成了以下三种优化方法:

| 优化方法 | 说明 | 效果 |
|---------|------|------|
| **Flash Attention 2** | 内存高效的注意力计算 | 降低显存占用，加速训练 |
| **Gradient Checkpointing** | 梯度检查点 | 减少显存消耗 |
| **Fused AdamW** | 融合的 AdamW 优化器 | 提高训练速度 |

## 与 Uni-LoRA 的区别

| 特性 | Standard LoRA | Uni-LoRA |
|------|--------------|----------|
| 实现方式 | PEFT 库 (HuggingFace) | 自定义实现 |
| 参数共享 | 无共享向量 | 共享向量银行 |
| 适用层 | 可配置任意目标模块 | MoE 专家层 + Attention |
| 量化支持 | QLoRA (4/8-bit) | 自定义量化 |
| 保存格式 | PEFT 标准格式 | 自定义格式 |

## 快速开始

### 1. 安装依赖

```bash
pip install peft>=0.7.0 bitsandbytes>=0.41.0
```

### 2. 训练

**全精度训练 (BF16):**
```bash
cd training
bash run_standard_lora.sh
```

**4-bit 量化训练 (QLoRA):**
```bash
cd training
bash run_standard_lora_4bit.sh
```

### 3. 测试

**交互式模式:**
```bash
cd testing
python test_standard_lora.py \
    --base-model /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --adapter-path ../training/output/standard_lora \
    --interactive
```

**批量测试:**
```bash
python test_standard_lora.py \
    --base-model /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --adapter-path ../training/output/standard_lora \
    --test-prompts-file test_prompts.json \
    --output-dir ./results
```

## 可配置参数

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_r` | 64 | LoRA 秩 |
| `--lora_alpha` | 16 | LoRA alpha 缩放因子 |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--target_modules` | q,k,v,o,gate,up,down | 目标模块 |

### 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bits` | 16 | 量化位数 (4/8/16) |
| `--double_quant` | True | 双重量化 |
| `--quant_type` | nf4 | 量化类型 |

## 目录结构

```
qwen_moe_standard_lora/
├── README.md
├── training/
│   ├── train_standard_lora.py      # 训练脚本
│   ├── run_standard_lora.sh        # 全精度训练
│   └── run_standard_lora_4bit.sh   # 4-bit QLoRA 训练
└── testing/
    └── test_standard_lora.py       # 测试脚本
```

## 重要提示：Prompt 格式问题

### 问题说明

如果模型在推理时只能对问题进行补全而不是回答，这通常是因为**训练和推理时使用的 prompt 格式不一致**导致的。

**训练时使用的格式**（Alpaca 格式）：
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{你的问题}

### Response: 
```

**推理时如果直接使用原始问题**：
```
{你的问题}
```

模型会认为这是一个需要补全的文本，而不是一个需要回答的指令。

### 解决方案

测试脚本已经默认启用了 Alpaca 格式（`--use-alpaca-format`，默认开启），会自动将你的问题格式化为训练时的格式。

**使用示例**：
```bash
# 交互式测试（自动使用 Alpaca 格式）
python test_standard_lora.py \
    --base-model /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --adapter-path ../training/output/standard_lora \
    --interactive

# 如果不想使用 Alpaca 格式（不推荐）
python test_standard_lora.py \
    --base-model /path/to/Qwen1.5-MoE-A2.7B-Chat \
    --adapter-path ../training/output/standard_lora \
    --interactive \
    --no-alpaca-format
```

### 自定义 Prompt 格式

如果你在训练时使用了不同的 prompt 格式，需要修改 `test_standard_lora.py` 中的 `format_alpaca_prompt` 函数，使其与训练时的格式保持一致。

## 注意事项

1. **显存需求**: 全精度训练需要约 40GB 显存，4-bit 量化训练约需 20GB
2. **PEFT 版本**: 建议使用 peft>=0.7.0 以获得最佳兼容性
3. **模型合并**: 训练完成后可使用 `model.merge_and_unload()` 合并权重
4. **Prompt 格式**: 推理时必须使用与训练时相同的 prompt 格式，否则模型可能只进行补全而不是回答
