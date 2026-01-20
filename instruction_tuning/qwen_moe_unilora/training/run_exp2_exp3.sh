#!/bin/bash
# Run exp2 and exp3 experiments only (exp1 already completed)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate instruction_tuning
set -euo pipefail

PYTHON=/root/miniconda3/envs/instruction_tuning/bin/python
TRAINING_DIR=/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/training
TESTING_DIR=/root/autodl-tmp/Uni-LoRA/instruction_tuning/qwen_moe_unilora/testing
BASE_MODEL="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
TEST_PROMPTS="${TESTING_DIR}/test_prompts_tcmqa.json"
OUTPUT_BASE="${TESTING_DIR}/test_results_combine"

mkdir -p ${OUTPUT_BASE}

echo "=============================================="
echo "Starting Experiments 2 & 3 (exp1 already done)"
echo "Time: $(date)"
echo "=============================================="

# Function to test a trained model
test_model() {
    local exp_name=$1
    local model_path=$2
    local output_dir="${OUTPUT_BASE}/${exp_name}"
    
    echo ">>> Testing ${exp_name}..."
    mkdir -p ${output_dir}
    
    cd ${TESTING_DIR}
    ${PYTHON} test_model_with_fastchat.py \
        --base-model ${BASE_MODEL} \
        --trained-model ${model_path} \
        --test-prompts-file ${TEST_PROMPTS} \
        --output-dir ${output_dir} \
        --mode single \
        2>&1 | tee ${output_dir}/test.log
    
    echo "✅ ${exp_name} testing completed!"
}

# ==========================================
# Experiment 2: Attention Only
# ==========================================
echo ""
echo ">>> [EXPERIMENT 2] Attention Only..."
echo "Start time: $(date)"
cd ${TRAINING_DIR}
bash run_exp2_attn_only.sh

# Test Experiment 2
test_model "exp2_attn_only" "${TRAINING_DIR}/output/exp2_attn_only"

# ==========================================
# Experiment 3: Full Combined
# ==========================================
echo ""
echo ">>> [EXPERIMENT 3] Full Combined..."
echo "Start time: $(date)"
cd ${TRAINING_DIR}
bash run_exp3_full.sh

# Test Experiment 3
test_model "exp3_full_combined" "${TRAINING_DIR}/output/exp3_full_combined"

echo ""
echo "=============================================="
echo "🎉 Experiments 2 & 3 Completed!"
echo "End time: $(date)"
echo "=============================================="
echo "Results saved to: ${OUTPUT_BASE}"
