#!/bin/bash

# run_unilora_pipeline_0117.sh
# Optimized Uni-LoRA + MoE training with JOINT TRAINING mode
# Based on analysis report recommendations:
# - Joint training (stage=0) instead of alternating
# - Load balancing loss enabled
# - 3 epochs training
# - Flexible shared vector rank

# === Environment Activation ===
source /root/miniconda3/etc/profile.d/conda.sh
conda activate instruction_tuning

set -euo pipefail

# Configure proxy settings to bypass proxy for SwanLab
export no_proxy="${no_proxy:-},swanlab.cn,api.swanlab.cn"

# Python executable
PYTHON=/root/miniconda3/envs/instruction_tuning/bin/python

# === Configuration ===
MODEL_NAME=${1:-"/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"}
DATASET=${2:-"/root/autodl-tmp/data/fomat/qa_alpaca_clean.jsonl"}
OUTPUT_BASE_DIR=${3:-"./output/qwen_moe_unilora_pipeline_0117"}

# Uni-LoRA Configuration (OPTIMIZED)
LORA_R=64
LORA_ALPHA=16.0
USE_RANK1=False
SHARED_VECTOR_RANK=8  # Flexible shared vector rank (not fixed to 1)
BITS=16

# ============================================================
# TRAINING HYPERPARAMETERS (OPTIMIZED FOR QUALITY)
# ============================================================
NUM_GPUS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8

# Learning rates (differential)
LEARNING_RATE=5e-5
LEARNING_RATE_VECTOR_BANK=1e-3
LEARNING_RATE_ROUTER=1e-5  # Lower LR for router

# Training duration
NUM_TRAIN_EPOCHS=3  # Increased from 1 to 3
SAVE_STEPS=500
EVAL_STEPS=500
MAX_STEPS=-1
WARMUP_RATIO=0.03

# Load balancing
LOAD_BALANCING_COEF=0.01

MAX_GRAD_NORM=1.0
LR_SCHEDULER_TYPE="cosine"

# Use GPU 0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Single GPU training mode - DeepSpeed disabled"

# Environment Setup
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Starting Uni-LoRA Pipeline (Optimized 0117)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_BASE_DIR"
echo "LoRA Rank: $LORA_R, Alpha: $LORA_ALPHA"
echo ""
echo "[OPTIMIZED TRAINING PARAMETERS]"
echo "  Training Mode: JOINT (stage=0)"
echo "  Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Save/Eval Steps: $SAVE_STEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Router Learning Rate: $LEARNING_RATE_ROUTER"
echo "  Vector Bank LR: $LEARNING_RATE_VECTOR_BANK"
echo "  Shared Vector Rank: $SHARED_VECTOR_RANK"
echo "  Load Balancing Coef: $LOAD_BALANCING_COEF"
echo "  Precision: BF16"
echo "  [SwanLab monitoring ENABLED]"
echo "=============================================="

mkdir -p ${OUTPUT_BASE_DIR}

# === Joint Training (Combined Router + Adapters) ===
echo ""
echo ">>> [JOINT TRAINING] Training Router and Uni-LoRA Adapters together..."
echo ""

OUTPUT_DIR="${OUTPUT_BASE_DIR}/joint"

${PYTHON} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --dataset_format alpaca \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_vector_bank ${LEARNING_RATE_VECTOR_BANK} \
    --learning_rate_router ${LEARNING_RATE_ROUTER} \
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
    --training_stage 0 \
    --shared_vector_rank ${SHARED_VECTOR_RANK} \
    --load_balancing_coef ${LOAD_BALANCING_COEF} \
    --bits ${BITS} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to swanlab \
    2>&1 | tee ${OUTPUT_BASE_DIR}/joint_training.log

TRAIN_STATUS=${PIPESTATUS[0]}
if [ ${TRAIN_STATUS} -ne 0 ]; then
    echo "❌ Joint Training Failed! Exit Code: ${TRAIN_STATUS}"
    exit ${TRAIN_STATUS}
fi

echo "✅ Joint Training Completed Successfully!"
echo "🎉 Uni-LoRA Optimized Pipeline Finished!"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Key optimizations applied:"
echo "  ✓ Joint training (Router + Adapters together)"
echo "  ✓ Load balancing loss (coef=${LOAD_BALANCING_COEF})"
echo "  ✓ 3 epochs training"
echo "  ✓ Flexible shared vector rank (${SHARED_VECTOR_RANK})"
echo "  ✓ Differential learning rates (Router: ${LEARNING_RATE_ROUTER})"
