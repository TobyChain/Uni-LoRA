# 文件结构说明

## 📂 目录组织

所有与 Qwen MoE + Uni-LoRA 相关的文件已整理到 `qwen_moe_unilora/` 目录下，按功能分类：

```
qwen_moe_unilora/
├── README.md                    # 主 README（快速开始指南）
├── STRUCTURE.md                 # 本文件（结构说明）
├── .gitignore                   # Git 忽略文件
├── run_training.sh              # 便捷训练启动脚本
├── run_testing.sh               # 便捷测试启动脚本
│
├── modeling/                    # 模型实现
│   └── modeling_unilora_moe.py # Uni-LoRA MoE 核心实现
│
├── training/                    # 训练相关
│   ├── train_unilora_moe.py    # 训练脚本（Python）
│   └── finetune_qwem1.5_moe_A2.7_unilora_1229.sh  # 训练启动脚本（Shell）
│
├── configs/                     # 配置文件
│   ├── ds_config.json          # DeepSpeed ZeRO-2 配置
│   └── ds_config_zero3.json   # DeepSpeed ZeRO-3 配置
│
├── testing/                     # 测试相关
│   ├── test_model_with_fastchat.py  # FastChat 测试脚本
│   ├── test_model.sh           # 测试启动脚本
│   └── test_prompts.json       # 测试提示集合
│
└── docs/                        # 文档
    ├── README_unilora_moe.md   # 主要文档（详细说明）
    └── README_TESTING.md       # 测试文档
```

## 🔗 路径关系

### 训练脚本路径
- 从 `training/` 目录运行：
  - DeepSpeed 配置：`../configs/ds_config.json`
  - 模型代码：自动通过 `sys.path` 添加 `../modeling/`

### 测试脚本路径
- 从 `testing/` 目录运行：
  - 训练输出：`../output/qwen_moe_unilora/stage1`
  - 测试结果：`../test_results`

### 便捷脚本
- 从 `qwen_moe_unilora/` 根目录运行：
  - `./run_training.sh` → 进入 `training/` 并运行训练
  - `./run_testing.sh` → 进入 `testing/` 并运行测试

## 📝 文件说明

### 核心实现
- **modeling/modeling_unilora_moe.py**: 
  - `UniLoRALinear`: Uni-LoRA 线性层
  - `UniLoRAMoEExpert`: MoE 专家包装器
  - `UniLoRAMoELayer`: MoE 层包装器
  - `apply_unilora_to_qwen_moe()`: 应用函数
  - `freeze_router()` / `freeze_unilora_adapters()`: 冻结函数

### 训练脚本
- **training/train_unilora_moe.py**: 
  - 完整的训练流程
  - 支持两阶段训练
  - 4-bit 量化加载
  - DeepSpeed 集成

- **training/finetune_qwem1.5_moe_A2.7_unilora_1229.sh**:
  - 训练启动脚本
  - 可配置超参数
  - 自动检查点管理

### 配置文件
- **configs/ds_config.json**: ZeRO-2 配置（推荐）
- **configs/ds_config_zero3.json**: ZeRO-3 配置（内存优化）

### 测试脚本
- **testing/test_model_with_fastchat.py**: 
  - FastChat 集成测试
  - 模型对比功能
  - 结果保存

- **testing/test_model.sh**: 
  - 一键测试脚本
  - 自动查找检查点

- **testing/test_prompts.json**: 
  - 预定义测试提示
  - 支持自定义

### 文档
- **README.md**: 快速开始指南
- **docs/README_unilora_moe.md**: 详细使用文档
- **docs/README_TESTING.md**: 测试指南

## 🚀 使用方式

### 方式 1: 使用便捷脚本（推荐）
```bash
cd qwen_moe_unilora

# 训练
./run_training.sh

# 测试
./run_testing.sh
```

### 方式 2: 直接运行
```bash
# 训练
cd training
bash finetune_qwem1.5_moe_A2.7_unilora_1229.sh

# 测试
cd testing
bash test_model.sh
```

### 方式 3: Python 直接调用
```bash
# 训练
cd training
python train_unilora_moe.py --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B-Chat ...

# 测试
cd testing
python test_model_with_fastchat.py --base-model ... --trained-model ...
```

## ⚠️ 注意事项

1. **路径依赖**: 脚本设计为从各自目录运行，确保路径正确
2. **导入路径**: `train_unilora_moe.py` 会自动添加 `modeling/` 到 Python 路径
3. **配置文件**: DeepSpeed 配置使用相对路径 `../configs/`
4. **输出目录**: 训练输出在 `../output/`，测试结果在 `../test_results/`

## 📦 文件迁移历史

以下文件已从 `instruction_tuning/` 根目录移动到 `qwen_moe_unilora/`:

- `modeling_unilora_moe.py` → `modeling/modeling_unilora_moe.py`
- `train_unilora_moe.py` → `training/train_unilora_moe.py`
- `finetune_qwem1.5_moe_A2.7_unilora_1229.sh` → `training/finetune_qwem1.5_moe_A2.7_unilora_1229.sh`
- `ds_config.json` → `configs/ds_config.json`
- `ds_config_zero3.json` → `configs/ds_config_zero3.json`
- `test_model_with_fastchat.py` → `testing/test_model_with_fastchat.py`
- `test_model.sh` → `testing/test_model.sh`
- `test_prompts.json` → `testing/test_prompts.json`
- `README_unilora_moe.md` → `docs/README_unilora_moe.md`
- `README_TESTING.md` → `docs/README_TESTING.md`

## 🔄 后续维护

- 新增模型实现 → `modeling/`
- 新增训练脚本 → `training/`
- 新增配置文件 → `configs/`
- 新增测试脚本 → `testing/`
- 新增文档 → `docs/`

