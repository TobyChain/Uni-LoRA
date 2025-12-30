# 模型测试指南

使用 FastChat 测试训练前后的 Qwen MoE 模型效果。

## 快速开始

### 1. 修改数据集为 alpaca-clean

训练脚本已更新，默认使用 `alpaca-clean` 数据集：

```bash
# 在 finetune_qwem1.5_moe_A2.7_unilora_1229.sh 中
DATASET="alpaca-clean"  # 已修改
```

### 2. 运行训练

```bash
cd /data4/guanbingtao/Uni-LoRA/instruction_tuning
bash finetune_qwem1.5_moe_A2.7_unilora_1229.sh
```

### 3. 测试模型

#### 方法 1: 使用自动化测试脚本（推荐）

```bash
# 修改 test_model.sh 中的模型路径
TRAINED_MODEL_DIR="./output/qwen_moe_unilora/stage1"  # 或 stage2

# 运行测试
bash test_model.sh
```

#### 方法 2: 使用 Python 脚本

```bash
python test_model_with_fastchat.py \
    --base-model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --trained-model ./output/qwen_moe_unilora/stage1/checkpoint-XXX \
    --test-prompts-file test_prompts.json \
    --output-dir ./test_results \
    --mode compare
```

#### 方法 3: 使用 FastChat CLI（交互式）

```bash
# 测试原始模型
python -m fastchat.serve.cli \
    --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat

# 测试训练后模型
python -m fastchat.serve.cli \
    --model-path ./output/qwen_moe_unilora/stage1/checkpoint-XXX
```

## 测试提示文件格式

`test_prompts.json` 格式：

```json
[
    {
        "prompt": "解释一下什么是机器学习。",
        "category": "知识问答"
    },
    {
        "prompt": "写一个 Python 函数来计算斐波那契数列。",
        "category": "代码生成"
    }
]
```

## 输出结果

测试完成后，会在 `./test_results/` 目录生成：

- `base_model_results.json`: 原始模型的响应
- `trained_model_results.json`: 训练后模型的响应
- `comparison.json`: 对比结果

## 对比分析

`comparison.json` 包含每个提示的两个模型响应，方便进行对比分析：

```json
[
    {
        "prompt": "解释一下什么是机器学习。",
        "base_model_response": "...",
        "trained_model_response": "..."
    }
]
```

## 注意事项

1. **模型路径**: 确保训练后的模型路径正确
2. **内存要求**: 测试需要加载两个模型，确保有足够内存
3. **GPU**: 如果有 GPU，会自动使用 GPU 加速
4. **FastChat 安装**: 确保已安装 FastChat
   ```bash
   pip install fschat
   ```

## 自定义测试提示

编辑 `test_prompts.json` 或使用命令行参数：

```bash
python test_model_with_fastchat.py \
    --base-model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --trained-model ./output/... \
    --test-prompts "提示1" "提示2" "提示3"
```

## 故障排除

### FastChat 未安装
```bash
pip install fschat
# 或
cd /data4/guanbingtao/FastChat
pip install -e .
```

### 模型加载失败
- 检查模型路径是否正确
- 确保模型文件完整
- 检查是否有足够的 GPU 内存

### 测试超时
- 增加超时时间
- 减少测试提示数量
- 使用更小的模型进行测试

