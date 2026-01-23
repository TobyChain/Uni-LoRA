# Single GPU A100-40GB Training Setup

## 修改总结 (Summary of Changes)

本配置已针对**单卡 A100-PCIE-40G**进行优化，使用**8-bit量化**和**Uni-LoRA**方法进行分步训练。

### 主要修改 (Key Changes)

#### 1. 硬件配置 (Hardware Configuration)
- **GPU数量**: 2 → **1** (单卡训练)
- **量化方式**: 16-bit → **8-bit** (节省显存)
- **DeepSpeed**: 已禁用 (单卡不需要分片)

#### 2. 训练参数调整 (Training Parameters)

| 参数 | 原值 (2x24GB) | 新值 (1x40GB) |
|------|---------------|---------------|
| NUM_GPUS | 2 | **1** |
| BITS | 16 | **8** |
| PER_DEVICE_TRAIN_BATCH_SIZE | 1 | **4** |
| GRADIENT_ACCUMULATION_STEPS | 24 | **8** |
| 有效批次大小 | 48 | **32** |

**注意**: 有效批次大小 = `PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × NUM_GPUS`

#### 3. 文件修改 (Modified Files)

**`run_unilora_pipeline.sh`**:
- ✅ 设置 `BITS=8` 启用8-bit量化
- ✅ 设置 `NUM_GPUS=1` 单卡训练
- ✅ 调整批次大小为 `PER_DEVICE_TRAIN_BATCH_SIZE=4`
- ✅ 调整梯度累积为 `GRADIENT_ACCUMULATION_STEPS=8`
- ✅ 移除DeepSpeed启动器，使用直接Python执行
- ✅ 移除 `--deepspeed` 参数

**`train_unilora_moe.py`**:
- ✅ 无需修改 (已支持8-bit量化和单卡训练)

## 使用方法 (Usage)

### 1. 启动训练 (Start Training)

```bash
# 从项目根目录运行
cd /root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora

# 前台运行
bash run_training.sh

# 后台运行
nohup bash run_training.sh > training.log 2>&1 &
```

### 2. 自定义参数 (Custom Parameters)

```bash
cd training
bash run_unilora_pipeline.sh \
    "Qwen/Qwen1.5-MoE-A2.7B-Chat" \
    "alpaca-clean" \
    "./output/my_custom_output"
```

### 3. 指定GPU (Specify GPU)

```bash
# 使用GPU 0
CUDA_VISIBLE_DEVICES=0 bash run_training.sh

# 脚本会自动选择最空闲的GPU（如果未设置CUDA_VISIBLE_DEVICES）
```

## 训练流程 (Training Pipeline)

训练分为两个阶段:

### Stage 1: 训练Uni-LoRA适配器 (Train Adapters)
- **冻结**: Router (路由器)
- **训练**: Uni-LoRA adapters (专家适配器)
- **输出**: `./output/qwen_moe_unilora_pipeline/stage1/`

### Stage 2: 训练Router (Train Router)
- **冻结**: Uni-LoRA adapters (专家适配器)
- **训练**: Router (路由器)
- **加载**: Stage 1的适配器参数
- **输出**: `./output/qwen_moe_unilora_pipeline/stage2/`

## 显存使用估算 (Memory Usage Estimation)

使用8-bit量化在A100-40GB上训练Qwen1.5-MoE-A2.7B:

- **模型加载** (8-bit): ~3-4 GB
- **Uni-LoRA参数**: ~500 MB
- **激活值 + 梯度** (batch_size=4): ~8-12 GB
- **优化器状态**: ~2-3 GB
- **总计**: ~15-20 GB (留有充足余量)

## 配置详情 (Configuration Details)

### Uni-LoRA参数
- **Rank (r)**: 64
- **Alpha**: 16.0
- **Use Rank-1**: True (使用rank-1共享向量)
- **Dropout**: 0.0

### 优化器配置
- **学习率 (LR)**: 2e-4
- **Vector Bank LR**: 1e-3 (共享向量库使用更高学习率)
- **Warmup Ratio**: 0.03
- **Max Grad Norm**: 0.3
- **训练轮数**: 3 epochs

### 数据配置
- **默认数据集**: alpaca-clean
- **Source Max Length**: 1024
- **Target Max Length**: 256

## 监控训练 (Monitor Training)

```bash
# 查看训练日志
tail -f output/qwen_moe_unilora_pipeline/stage1_training.log
tail -f output/qwen_moe_unilora_pipeline/stage2_training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

## 故障排除 (Troubleshooting)

### 显存不足 (OOM)
如果遇到显存不足，可以调整以下参数:

```bash
# 在 run_unilora_pipeline.sh 中修改
PER_DEVICE_TRAIN_BATCH_SIZE=2  # 减小批次大小
GRADIENT_ACCUMULATION_STEPS=16  # 增加梯度累积
```

### 加载模型失败
确保模型已下载到正确位置:
```bash
# 检查模型路径
ls -la ./model/Qwen/Qwen1.5-MoE-A2.7B-Chat/
```

## 性能优化建议 (Performance Tips)

1. **使用bf16**: 已启用 `--bf16` 标志
2. **梯度检查点**: 已启用 `--gradient_checkpointing` 减少显存
3. **按长度分组**: 已启用 `--group_by_length` 提高效率
4. **8-bit量化**: 使用BitsAndBytes进行高效量化

## 输出文件 (Output Files)

训练完成后，输出目录包含:

```
output/qwen_moe_unilora_pipeline/
├── stage1/
│   ├── checkpoint-XXX/
│   │   └── adapter_model/
│   │       └── unilora_params.pt  # Uni-LoRA参数
│   ├── trainer_state.json
│   └── training_args.bin
├── stage2/
│   ├── checkpoint-XXX/
│   │   └── adapter_model/
│   │       └── unilora_params.pt
│   └── ...
├── stage1_training.log
└── stage2_training.log
```

## 技术细节 (Technical Details)

### 8-bit量化实现
使用 `BitsAndBytesConfig`:
```python
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

### 设备映射 (Device Map)
单GPU自动使用:
```python
device_map = "auto"  # 自动分配到可用GPU
```

### Uni-LoRA架构
- 共享向量库 (Shared Vector Bank)
- 专家特定投影 (Expert-specific Projections)
- 两阶段训练策略 (Two-stage Training)

---

**最后更新**: 2026-01-06
**适用模型**: Qwen/Qwen1.5-MoE-A2.7B-Chat
**硬件要求**: 1x A100-PCIE-40G (或类似40GB+ GPU)
