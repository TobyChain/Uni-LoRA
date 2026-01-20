#!/bin/bash

# 使用 FastChat 测试训练前后的模型效果 (TCMQA Dataset)

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 配置
BASE_MODEL="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
TRAINED_MODEL_DIR="../training/output/qwen_moe_unilora_pipeline_0114/stage2"
TEST_PROMPTS_FILE="test_prompts_tcmqa.json"
OUTPUT_DIR="./test_results_tcmqa"

# 修复 bitsandbytes 找不到 cuda 库的问题
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 检查训练后的模型路径
if [ ! -d "${TRAINED_MODEL_DIR}" ]; then
    echo "错误: 训练后的模型目录不存在: ${TRAINED_MODEL_DIR}"
    echo "请检查训练是否完成"
    exit 1
fi

# 检查是否包含 Uni-LoRA 参数
if [ ! -f "${TRAINED_MODEL_DIR}/unilora_params.pt" ] && [ ! -f "${TRAINED_MODEL_DIR}/adapter_model/unilora_params.pt" ]; then
    echo "警告: 在 ${TRAINED_MODEL_DIR} 中未找到 unilora_params.pt"
    # 尝试查找其他 checkpoint
    CHECKPOINT=$(find ${TRAINED_MODEL_DIR} -name "checkpoint-*" -type d | sort -rV | head -n 1)
    if [ -n "$CHECKPOINT" ]; then
        echo "发现检查点: $CHECKPOINT"
        TRAINED_MODEL_DIR=$CHECKPOINT
    else
        echo "未找到可用检查点，将尝试直接加载目录"
    fi
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "TCMQA 模型评估"
echo "=========================================="
echo "原始模型: ${BASE_MODEL}"
echo "训练后模型: ${TRAINED_MODEL_DIR}"
echo "测试提示文件: ${TEST_PROMPTS_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 运行测试
/root/miniconda3/envs/instruction_tuning/bin/python test_model_with_fastchat.py \
    --base-model ${BASE_MODEL} \
    --trained-model ${TRAINED_MODEL_DIR} \
    --test-prompts-file ${TEST_PROMPTS_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --mode compare

echo ""
echo "测试完成！结果保存在: ${OUTPUT_DIR}"
