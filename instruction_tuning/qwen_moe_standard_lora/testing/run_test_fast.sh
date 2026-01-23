#!/bin/bash
# Test Standard LoRA model with TCMQA prompts - FAST MODE (4-bit Batch 8)
# Uses 4-bit quantization (memory efficient) and Batch Size 8 (throughput)

source /root/miniforge3/etc/profile.d/conda.sh
conda activate instruction_tuning
set -euo pipefail

PYTHON=/root/miniforge3/envs/instruction_tuning/bin/python
BASE_MODEL="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
ADAPTER_PATH="../training/output/standard_lora/checkpoint-2000"
TEST_PROMPTS="./test_prompts_tcmqa.json"
OUTPUT_DIR="./test_results_tcmqa_fast"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Testing Standard LoRA Model (FAST MODE - 4bit BS8) ==="
echo "Base Model: $BASE_MODEL"
echo "Adapter: $ADAPTER_PATH"
echo "Test Prompts: $TEST_PROMPTS"
echo "Output: $OUTPUT_DIR"
echo "Batch Size: 8"
echo "Precision: 4-bit (bits=4)"
echo ""

${PYTHON} test_standard_lora.py \
    --base-model ${BASE_MODEL} \
    --adapter-path ${ADAPTER_PATH} \
    --test-prompts-file ${TEST_PROMPTS} \
    --output-dir ${OUTPUT_DIR} \
    --bits 4 \
    --batch-size 8 \
    --max-new-tokens 512 \
    --temperature 0.7

echo ""
echo "✅ Fast Testing Completed! Check results in: $OUTPUT_DIR/test_results.json"
