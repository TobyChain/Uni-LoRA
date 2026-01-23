#!/bin/bash
# Test script for Standard LoRA model
# Uses nohup to run in background

source /root/miniforge3/etc/profile.d/conda.sh
conda activate instruction_tuning
set -euo pipefail

PYTHON=/root/miniforge3/envs/instruction_tuning/bin/python
BASE_MODEL="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
ADAPTER_PATH="../training/output/standard_lora"
OUTPUT_DIR="./test_results"
LOG_FILE="./test_results/test.log"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=== Testing Standard LoRA Model ==="
echo "Base Model: ${BASE_MODEL}"
echo "Adapter Path: ${ADAPTER_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo ""

# Run test in demo mode (you can modify this to use --interactive or --test-prompts-file)
${PYTHON} test_standard_lora.py \
    --base-model ${BASE_MODEL} \
    --adapter-path ${ADAPTER_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --bits 16 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    2>&1 | tee ${LOG_FILE}

echo "✅ Testing Completed! Results saved to ${OUTPUT_DIR}"
