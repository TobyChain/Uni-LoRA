# Uni-LoRA MoE 训练策略分析报告

## 概述

本报告分析了当前 Uni-LoRA 两阶段训练策略在 Qwen MoE 模型微调中未能取得显著效果的原因，并提供基于 MoE 设计原理的优化建议。

## 当前训练策略回顾

### 两阶段训练流程

```
Stage 1: 训练专家适配器 (Adapters)
├── 冻结: Router (Gate)
├── 训练: Uni-LoRA 投影矩阵 + 共享向量
└── 学习率: 1e-4

Stage 2: 训练路由器 (Router)
├── 冻结: Uni-LoRA 适配器
├── 训练: Gate 权重
└── 学习率: 5e-5
```

### 关键参数配置
| 参数 | Stage 1 | Stage 2 |
|------|---------|---------|
| 学习率 | 1e-4 | 5e-5 |
| Epochs | 1 | 1 |
| Batch Size | 8 | 8 |
| LoRA Rank | 64 | 64 |

---

## 问题分析

### 1. 交替训练 vs 联合训练

**核心问题：路由器-专家协同适应被阻断**

MoE 模型的核心机制是：
- **Router (路由器)**: 决定将哪些 token 分配给哪些专家
- **Experts (专家)**: 根据分配处理对应的 token

> [!WARNING]
> 交替训练的根本问题在于：**Router 和 Experts 无法同时学习如何配合工作**

研究表明，联合训练允许 Router 和 Experts **共同进化**[1]。当 Router 被冻结时：
- Stage 1 中的专家只能适应原始（预训练）的路由策略
- Stage 2 中的 Router 需要学习路由到"已固化"的适配器

这导致了**策略不匹配**：专家为旧路由优化，而新路由指向的专家可能并非最优。

### 2. 缺乏负载均衡损失 (Load Balancing Loss)

当前实现中**没有辅助负载均衡损失**。

```python
# train_unilora_moe.py 中没有发现 auxiliary_loss 或 load_balancing_loss
```

MoE 训练的关键挑战是**专家崩塌 (Expert Collapse)**：
- 少数专家被过度使用，其他专家闲置
- 导致模型容量未被充分利用

标准做法是添加辅助损失：

```
L_total = L_task + α * L_balance

其中 L_balance = Σ f_i * P_i  (强制专家均匀分布)
```

### 3. Stage 2 未正确继承训练状态

当前代码逻辑（`train_unilora_moe.py:802-820`）：
```python
if unilora_args.training_stage == 2:
    # 仅加载适配器参数，未考虑 Router 的初始化
    unilora_params = torch.load(unilora_params_path)
    # Router 仍使用预训练权重，而非任何学习后的状态
```

> [!CAUTION]
> Stage 2 从头开始训练 Router，但 Router 需要理解 Stage 1 中适配器学到的表示

### 4. 训练数据量与 Epochs 不足

- **数据量**: 117,740 条
- **Epochs**: 仅 1 个
- **每个 Stage 独立训练**: 两阶段各 1 epoch

对于 MoE 这种高容量模型，单 epoch 训练可能不足以让专家充分特化。

### 5. Rank-1 向量共享的局限性

当前使用 `use_rank1=True`：
```python
# 所有专家共享同一个向量，仅通过投影矩阵区分
shared_vector: [2048, 1]  # 全模型共享
projection: [out_features, 1]  # 每个专家独立
```

这种极端参数高效的设计可能**限制了专家的差异化能力**。

---

## MoE 微调最佳实践

### 1. 使用联合训练而非交替训练
```python
# 推荐：同时训练 Router 和 Adapters
freeze_router(model, freeze=False)
freeze_unilora_adapters(model, freeze=False)
```

### 2. 添加负载均衡损失
```python
# 在 loss 计算中添加
router_logits = ...  # 从 gate 获取
balance_loss = compute_load_balancing_loss(router_logits)
total_loss = task_loss + 0.01 * balance_loss
```

### 3. 为 Router 使用更低的学习率
```python
# 差异化学习率
optimizer_groups = [
    {"params": expert_params, "lr": 1e-4},
    {"params": router_params, "lr": 1e-5},  # Router 更小
]
```

### 4. 增加训练 Epochs
- 医疗领域微调建议 3-5 epochs
- 使用 early stopping 防止过拟合

### 5. 监控 Router 行为
```python
# 记录每个专家的激活频率
gate_logits = self.gate(hidden_states)
expert_usage = (gate_logits.argmax(-1) == expert_id).sum()
```

---

## Uni-LoRA 优化方向

### 方向 1: 联合微调模式
```bash
# 修改 training_stage 逻辑，添加 stage=0 或 joint 模式
--training_stage 0  # 联合训练
```

### 方向 2: 增加 LoRA Rank
```bash
# 当前 rank=64，可提升至 128 或更高
--lora_r 128
--lora_alpha 32.0
```

### 方向 3: 逐步解冻策略 (Progressive Unfreezing)
```
Epoch 1-2: 仅训练适配器 (Router 冻结)
Epoch 3-5: 联合训练 (低学习率)
```

### 方向 4: 任务感知路由 (Task-Aware Gating)
为 Router 注入任务语义信息，使其了解当前处理的是医疗QA任务。

### 方向 5: 动态 Rank 分配 (DR-LoRA)
根据专家重要性动态调整 LoRA rank，高频激活的专家分配更多参数。

---

## 具体代码修改建议

### 1. 添加联合训练模式

在 `train_unilora_moe.py` 中：
```python
if unilora_args.training_stage == 0:  # 新增: 联合训练
    logger.info("Joint Training: Training both Router and Adapters")
    freeze_router(model, freeze=False)
    freeze_unilora_adapters(model, freeze=False)
```

### 2. 实现负载均衡损失

在 `modeling_unilora_moe.py` 的 `UniLoRAMoELayer.forward` 中：
```python
def forward(self, hidden_states, *args, **kwargs):
    # 计算路由权重
    router_logits = self.gate(hidden_states)
    
    # 负载均衡损失
    if self.training:
        balance_loss = self._compute_balance_loss(router_logits)
        # 存储供 Trainer 使用
        self.last_balance_loss = balance_loss
    
    # 继续正常前向传播...
```

### 3. 修改 Shell 脚本

```bash
# run_unilora_pipeline_joint.sh
${PYTHON} train_unilora_moe.py \
    --training_stage 0 \  # 联合训练
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --learning_rate_vector_bank 1e-4 \
    --learning_rate_router 1e-5 \  # 新增
```

---

## 结论

当前两阶段分离训练策略的核心问题是**阻断了 Router 和 Experts 的协同进化**。基于 MoE 设计原理，建议：

1. ✅ 切换到联合训练模式
2. ✅ 添加负载均衡辅助损失
3. ✅ 为 Router 使用差异化（更低）学习率
4. ✅ 增加训练 epochs 到 3-5
5. ✅ 监控专家激活分布

---

## 参考文献

1. Switch Transformers: Scaling to Trillion Parameter Models
2. MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based MoE
3. DR-LoRA: Dynamic Rank Allocation for Mixture of LoRA Experts
4. Mixture of LoRA Experts (MoLE)
5. TAG-MoE: Task-Aware Gating for MoE Fine-tuning

---

*报告生成时间: 2026-01-17*
*分析基于: run_unilora_pipeline_0114.sh, train_unilora_moe.py, modeling_unilora_moe.py*
