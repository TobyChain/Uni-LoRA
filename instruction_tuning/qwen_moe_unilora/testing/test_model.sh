#!/bin/bash

# 使用 FastChat 测试训练前后的模型效果

# 配置
BASE_MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat"
TRAINED_MODEL_DIR="../output/qwen_moe_unilora/stage1"  # 或 stage2
TEST_PROMPTS_FILE="test_prompts.json"
OUTPUT_DIR="../test_results"

# 检查训练后的模型路径
if [ ! -d "${TRAINED_MODEL_DIR}" ]; then
    echo "错误: 训练后的模型目录不存在: ${TRAINED_MODEL_DIR}"
    echo "请先完成训练，或修改 TRAINED_MODEL_DIR 变量"
    exit 1
fi

# 查找最佳检查点
CHECKPOINT=$(find ${TRAINED_MODEL_DIR} -name "checkpoint-*" -type d | sort -V | tail -1)

if [ -z "${CHECKPOINT}" ]; then
    echo "警告: 未找到检查点，使用模型目录: ${TRAINED_MODEL_DIR}"
    TRAINED_MODEL=${TRAINED_MODEL_DIR}
else
    echo "使用检查点: ${CHECKPOINT}"
    TRAINED_MODEL=${CHECKPOINT}
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "模型对比测试"
echo "=========================================="
echo "原始模型: ${BASE_MODEL}"
echo "训练后模型: ${TRAINED_MODEL}"
echo "测试提示文件: ${TEST_PROMPTS_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行测试
python test_model_with_fastchat.py \
    --base-model ${BASE_MODEL} \
    --trained-model ${TRAINED_MODEL} \
    --test-prompts-file test_prompts.json \
    --output-dir ${OUTPUT_DIR} \
    --mode compare

echo ""
echo "测试完成！结果保存在: ${OUTPUT_DIR}"

