#!/bin/bash

# Training script for Qwen MoE with Uni-LoRA
# Supports two-stage training with DeepSpeed ZeRO-2/3

# Configuration
MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B-Chat"
OUTPUT_DIR="./output/qwen_moe_unilora"
DATASET="alpaca-clean"
RANK=64
ALPHA=16.0
TRAINING_STAGE=1  # 1: train adapters, 2: train router

# DeepSpeed configuration (choose one)
# 注意: 从 training 目录运行时，需要相对路径
DS_CONFIG="../configs/ds_config.json"  # ZeRO-2
# DS_CONFIG="../configs/ds_config_zero3.json"  # ZeRO-3 with CPU offload

# Training hyperparameters
NUM_GPUS=4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=3
MAX_STEPS=-1
SAVE_STEPS=500
LOGGING_STEPS=10
EVAL_STEPS=500
WARMUP_RATIO=0.03
MAX_GRAD_NORM=1.0

# Dataset settings
SOURCE_MAX_LEN=1024
TARGET_MAX_LEN=256
MAX_TRAIN_SAMPLES=
MAX_EVAL_SAMPLES=

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Stage 1: Train Uni-LoRA adapters (freeze router)
if [ ${TRAINING_STAGE} -eq 1 ]; then
    echo "=========================================="
    echo "Stage 1: Training Uni-LoRA Adapters"
    echo "=========================================="
    
    deepspeed --num_gpus=${NUM_GPUS} train_unilora_moe.py \
        --model_name_or_path ${MODEL_NAME} \
        --dataset ${DATASET} \
        --output_dir ${OUTPUT_DIR}/stage1 \
        --deepspeed ${DS_CONFIG} \
        --do_train \
        --do_eval \
        --bf16 \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --max_steps ${MAX_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --logging_steps ${LOGGING_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --warmup_ratio ${WARMUP_RATIO} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --source_max_len ${SOURCE_MAX_LEN} \
        --target_max_len ${TARGET_MAX_LEN} \
        --rank ${RANK} \
        --alpha ${ALPHA} \
        --training_stage 1 \
        --gradient_checkpointing \
        --group_by_length \
        --save_strategy steps \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better false \
        --save_total_limit 3 \
        ${MAX_TRAIN_SAMPLES:+--max_train_samples ${MAX_TRAIN_SAMPLES}} \
        ${MAX_EVAL_SAMPLES:+--max_eval_samples ${MAX_EVAL_SAMPLES}} \
        2>&1 | tee ${OUTPUT_DIR}/stage1_training.log

    echo "Stage 1 training completed. Checkpoint saved at: ${OUTPUT_DIR}/stage1"
    
# Stage 2: Train router (freeze adapters)
elif [ ${TRAINING_STAGE} -eq 2 ]; then
    echo "=========================================="
    echo "Stage 2: Training Router"
    echo "=========================================="
    echo "Loading checkpoint from Stage 1: ${OUTPUT_DIR}/stage1"
    
    # Find the best checkpoint from stage 1
    STAGE1_CHECKPOINT=$(find ${OUTPUT_DIR}/stage1 -name "checkpoint-*" -type d | sort -V | tail -1)
    
    if [ -z "${STAGE1_CHECKPOINT}" ]; then
        echo "Error: No checkpoint found in ${OUTPUT_DIR}/stage1"
        echo "Please run Stage 1 training first."
        exit 1
    fi
    
    echo "Using checkpoint: ${STAGE1_CHECKPOINT}"
    
    deepspeed --num_gpus=${NUM_GPUS} train_unilora_moe.py \
        --model_name_or_path ${MODEL_NAME} \
        --dataset ${DATASET} \
        --output_dir ${OUTPUT_DIR}/stage2 \
        --deepspeed ${DS_CONFIG} \
        --do_train \
        --do_eval \
        --bf16 \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --max_steps ${MAX_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --logging_steps ${LOGGING_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --warmup_ratio ${WARMUP_RATIO} \
        --max_grad_norm ${MAX_GRAD_NORM} \
        --source_max_len ${SOURCE_MAX_LEN} \
        --target_max_len ${TARGET_MAX_LEN} \
        --rank ${RANK} \
        --alpha ${ALPHA} \
        --training_stage 2 \
        --gradient_checkpointing \
        --group_by_length \
        --save_strategy steps \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better false \
        --save_total_limit 3 \
        ${MAX_TRAIN_SAMPLES:+--max_train_samples ${MAX_TRAIN_SAMPLES}} \
        ${MAX_EVAL_SAMPLES:+--max_eval_samples ${MAX_EVAL_SAMPLES}} \
        2>&1 | tee ${OUTPUT_DIR}/stage2_training.log

    echo "Stage 2 training completed. Final checkpoint saved at: ${OUTPUT_DIR}/stage2"
    
else
    echo "Error: Invalid training stage: ${TRAINING_STAGE}. Must be 1 or 2."
    exit 1
fi

echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="

