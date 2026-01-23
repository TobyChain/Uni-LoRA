#!/bin/bash
# Test script for Standard LoRA model using TCMQA test prompts
# Uses nohup to run in background

source /root/miniforge3/etc/profile.d/conda.sh
conda activate instruction_tuning
set -euo pipefail

PYTHON=/root/miniforge3/envs/instruction_tuning/bin/python
BASE_MODEL="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
ADAPTER_PATH="../training/output/standard_lora"
TEST_PROMPTS_FILE="./test_prompts_tcmqa.json"
OUTPUT_DIR="./test_results_tcmqa"
LOG_FILE="./test_results_tcmqa/test.log"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=== Testing Standard LoRA Model with TCMQA Prompts ==="
echo "Base Model: ${BASE_MODEL}"
echo "Adapter Path: ${ADAPTER_PATH}"
echo "Test Prompts File: ${TEST_PROMPTS_FILE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo ""

# Count prompts
if [ -f "${TEST_PROMPTS_FILE}" ]; then
    PROMPT_COUNT=$(python3 -c "import json; print(len(json.load(open('${TEST_PROMPTS_FILE}'))))")
    echo "Found ${PROMPT_COUNT} test prompts"
    echo ""
fi

# Run test with test prompts file
${PYTHON} test_standard_lora.py \
    --base-model ${BASE_MODEL} \
    --adapter-path ${ADAPTER_PATH} \
    --test-prompts-file ${TEST_PROMPTS_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --bits 16 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --batch-size 4 \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "✅ Testing Completed! Results saved to ${OUTPUT_DIR}"
echo "Check ${LOG_FILE} for detailed logs"
echo "Check ${OUTPUT_DIR}/test_results.json for results"
