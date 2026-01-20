#!/bin/bash

# run_unilora_pipeline_optimized.sh
# Unified pipeline for Uni-LoRA + MoE training (Stage 1 + Stage 2)
# OPTIMIZED VERSION: Fixes training instability (sawtooth loss, oscillating grad norm)
#
# Key Optimization Changes (vs original):
# 1. Removed --group_by_length: Ensures random shuffle for I.I.D. mini-batches
# 2. MAX_GRAD_NORM: 0.3 -> 1.0 (standard value, prevents gradient signal masking)
# 3. LEARNING_RATE Stage1: 2e-4 -> 1e-4 (conservative LR for Uni-LoRA constraints)
# 4. Added --lr_scheduler_type cosine (smooth decay for better convergence)
# 5. Added --ddp_find_unused_parameters False (prevents Trainer internal issues)

# === Environment Activation ===
source /root/miniforge3/etc/profile.d/conda.sh
conda activate instruction_tuning

set -euo pipefail

# Configure proxy settings to bypass proxy for SwanLab (resolves connection issues)
export no_proxy="${no_proxy:-},swanlab.cn,api.swanlab.cn"

# === Configuration ===
MODEL_NAME=${1:-"/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"}
DATASET=${2:-"alpaca-clean"}
OUTPUT_BASE_DIR=${3:-"./output/qwen_moe_unilora_pipeline_0107"}

# Uni-LoRA Configuration (matching qlora_unilora.py defaults)
LORA_R=64
LORA_ALPHA=16.0
USE_RANK1=True  # Use rank-1 shared vector for memory efficiency
BITS=16  # 16-bit (BF16) for higher precision

# Training Hyperparameters (OPTIMIZED for stability)
NUM_GPUS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 4 * 8 = 32 (good)

# [OPTIMIZED] Learning rate: lowered from 2e-4 to 1e-4 for Stage 1 stability
LEARNING_RATE_S1=1e-4  # Conservative LR for adapter training
LEARNING_RATE_S2=5e-5  # Even lower for router fine-tuning (optional)
LEARNING_RATE_VECTOR_BANK=1e-3  # Keep vector bank LR unchanged

NUM_TRAIN_EPOCHS=3
SAVE_STEPS=500
EVAL_STEPS=500
MAX_STEPS=-1
WARMUP_RATIO=0.03

# [OPTIMIZED] Gradient clipping: 0.3 -> 1.0 (standard, prevents over-aggressive clipping)
MAX_GRAD_NORM=1.0

# LR Scheduler type for smooth convergence
LR_SCHEDULER_TYPE="cosine"

# Auto-select GPUs if CUDA_VISIBLE_DEVICES not set
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "Auto-selecting ${NUM_GPUS} most idle GPUs..."
    
    SELECTED_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while read line; do
        idx=$(echo $line | cut -d',' -f1)
        mem=$(echo $line | cut -d',' -f2)
        # Check if any compute or graphics process is running on this GPU index
        if ! nvidia-smi --query-compute-apps=gpu_index --format=csv,noheader | grep -q "^$idx$" && \
           ! nvidia-smi --query-accounted-apps=gpu_index --format=csv,noheader | grep -q "^$idx$"; then
            echo "$idx,$mem"
        fi
    done | sort -t',' -k2 -n | head -n ${NUM_GPUS} | cut -d',' -f1 | paste -sd ',' -)
    
    if [ -z "$SELECTED_GPUS" ]; then
        echo "Error: Failed to auto-select GPUs. Please set CUDA_VISIBLE_DEVICES manually."
        exit 1
    fi
    
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
    echo "Selected GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "Using pre-set GPUs: $CUDA_VISIBLE_DEVICES"
fi

# Single GPU training - no DeepSpeed
echo "Single GPU training mode - DeepSpeed disabled"

# Environment Setup
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Starting Uni-LoRA Pipeline (OPTIMIZED)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_BASE_DIR"
echo "LoRA Rank: $LORA_R, Alpha: $LORA_ALPHA"
echo ""
echo "[OPTIMIZED PARAMETERS]"
echo "  Stage 1 LR: $LEARNING_RATE_S1 (lowered from 2e-4)"
echo "  Stage 2 LR: $LEARNING_RATE_S2"
echo "  Vector Bank LR: $LEARNING_RATE_VECTOR_BANK"
echo "  Max Grad Norm: $MAX_GRAD_NORM (raised from 0.3)"
echo "  LR Scheduler: $LR_SCHEDULER_TYPE"
echo "  Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Data Shuffle: ENABLED (group_by_length removed)"
echo "=============================================="

mkdir -p ${OUTPUT_BASE_DIR}

# === Stage 1: Adapter Training (Router Frozen) ===
echo ""
echo ">>> [STAGE 1] Training Uni-LoRA Adapters (Experts)..."
echo ""

OUTPUT_DIR_S1="${OUTPUT_BASE_DIR}/stage1"

# Launch with python (single GPU, no DeepSpeed)
# [OPTIMIZED] Key changes:
#   - Removed --group_by_length (ensures random shuffle)
#   - Using lowered learning rate
#   - Added --lr_scheduler_type cosine
#   - Added --ddp_find_unused_parameters False
python train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR_S1} \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE_S1} \
    --learning_rate_vector_bank ${LEARNING_RATE_VECTOR_BANK} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --save_strategy steps \
    --eval_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --training_stage 1 \
    $([ "${USE_RANK1}" = "True" ] && echo "--use_rank1") \
    --bits ${BITS} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --save_total_limit 2 \
    --logging_steps 10 \
    --report_to none \
    2>&1 | tee ${OUTPUT_BASE_DIR}/stage1_training.log

ST1_STATUS=${PIPESTATUS[0]}
if [ ${ST1_STATUS} -ne 0 ]; then
    echo "❌ Stage 1 Failed! Exit Code: ${ST1_STATUS}"
    exit ${ST1_STATUS}
fi

echo "✅ Stage 1 Completed Successfully."

# === Stage 2: Router Training (Adapters Frozen) ===
echo ""
echo ">>> [STAGE 2] Training Router..."
echo ""

OUTPUT_DIR_S2="${OUTPUT_BASE_DIR}/stage2"

# [OPTIMIZED] Stage 2 uses even lower LR for router fine-tuning
python train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR_S2} \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE_S2} \
    --learning_rate_vector_bank ${LEARNING_RATE_VECTOR_BANK} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --save_strategy steps \
    --eval_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --training_stage 2 \
    $([ "${USE_RANK1}" = "True" ] && echo "--use_rank1") \
    --bits ${BITS} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --save_total_limit 2 \
    --logging_steps 10 \
    --report_to none \
    2>&1 | tee ${OUTPUT_BASE_DIR}/stage2_training.log

ST2_STATUS=${PIPESTATUS[0]}
if [ ${ST2_STATUS} -ne 0 ]; then
    echo "❌ Stage 2 Failed! Exit Code: ${ST2_STATUS}"
    exit ${ST2_STATUS}
fi

echo "✅ Stage 2 Completed Successfully."
echo "🎉 Full Uni-LoRA Pipeline Finished!"
echo ""
echo "Stage 1 output: ${OUTPUT_DIR_S1}"
echo "Stage 2 output: ${OUTPUT_DIR_S2}"
