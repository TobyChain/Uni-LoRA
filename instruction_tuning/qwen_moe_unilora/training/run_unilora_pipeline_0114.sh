#!/bin/bash

# run_unilora_pipeline_0114.sh
# Unified pipeline for Uni-LoRA + MoE training with Medical QA Dataset
# OPTIMIZED VERSION: SwanLab enabled, faster training, bf16 precision

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
OUTPUT_BASE_DIR=${3:-"./output/qwen_moe_unilora_pipeline_0114"}

# Uni-LoRA Configuration
LORA_R=64
LORA_ALPHA=16.0
USE_RANK1=True
BITS=16

# ============================================================
# TRAINING HYPERPARAMETERS (OPTIMIZED FOR SPEED)
# ============================================================
NUM_GPUS=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4

# Learning rates
LEARNING_RATE_S1=1e-4
LEARNING_RATE_S2=5e-5
LEARNING_RATE_VECTOR_BANK=1e-3

NUM_TRAIN_EPOCHS=1
SAVE_STEPS=1000
EVAL_STEPS=1000
MAX_STEPS=-1
WARMUP_RATIO=0.03

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
echo "Starting Uni-LoRA Pipeline (QA Dataset 0114)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_BASE_DIR"
echo "LoRA Rank: $LORA_R, Alpha: $LORA_ALPHA"
echo ""
echo "[OPTIMIZED TRAINING PARAMETERS]"
echo "  Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Save/Eval Steps: $SAVE_STEPS"
echo "  Stage 1 LR: $LEARNING_RATE_S1"
echo "  Stage 2 LR: $LEARNING_RATE_S2"
echo "  Precision: BF16"
echo "  [SwanLab monitoring ENABLED]"
echo "=============================================="

mkdir -p ${OUTPUT_BASE_DIR}

# === Stage 1: Adapter Training (Router Frozen) ===
echo ""
echo ">>> [STAGE 1] Training Uni-LoRA Adapters (Experts)..."
echo ""

OUTPUT_DIR_S1="${OUTPUT_BASE_DIR}/stage1"

${PYTHON} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --dataset_format alpaca \
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
    --report_to swanlab \
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

${PYTHON} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --dataset_format alpaca \
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
    --report_to swanlab \
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
