#!/bin/bash

# run_unilora_pipeline.sh
# Unified pipeline for Uni-LoRA + MoE training (Stage 1 + Stage 2)
# Refactored to match qlora_unilora.py design patterns

set -euo pipefail

# === Configuration ===
MODEL_NAME=${1:-"Qwen/Qwen1.5-MoE-A2.7B-Chat"}
DATASET=${2:-"alpaca-clean"}
OUTPUT_BASE_DIR=${3:-"./output/qwen_moe_unilora_pipeline"}

# Uni-LoRA Configuration (matching qlora_unilora.py defaults)
LORA_R=64
LORA_ALPHA=16.0
USE_RANK1=True  # Use rank-1 shared vector for memory efficiency
BITS=16  # No quantization (using bf16 + ZeRO-3 instead)

# Training Hyperparameters (optimized for 2x 24GB GPUs with ZeRO-3)
NUM_GPUS=2
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=24
LEARNING_RATE=2e-4
LEARNING_RATE_VECTOR_BANK=1e-3
NUM_TRAIN_EPOCHS=3
SAVE_STEPS=500
EVAL_STEPS=500
MAX_STEPS=-1
WARMUP_RATIO=0.03
MAX_GRAD_NORM=0.3

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

# DeepSpeed ZeRO-3 Config
DS_CONFIG="../configs/ds_config_zero3.json"
if [ ! -f "$DS_CONFIG" ]; then
    echo "Warning: DeepSpeed config not found at $DS_CONFIG. Checking current directory..."
    if [ -f "ds_config_zero3.json" ]; then
        DS_CONFIG="ds_config_zero3.json"
    else
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        if [ -f "$SCRIPT_DIR/../configs/ds_config_zero3.json" ]; then
             DS_CONFIG="$SCRIPT_DIR/../configs/ds_config_zero3.json"
        fi
    fi
fi
echo "Using DeepSpeed Config: $DS_CONFIG"

# Environment Setup
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Starting Uni-LoRA Pipeline (Refactored)"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_BASE_DIR"
echo "LoRA Rank: $LORA_R, Alpha: $LORA_ALPHA"
echo "LR: $LEARNING_RATE, Vector Bank LR: $LEARNING_RATE_VECTOR_BANK"
echo "=========================================="

mkdir -p ${OUTPUT_BASE_DIR}

# === Stage 1: Adapter Training (Router Frozen) ===
echo ""
echo ">>> [STAGE 1] Training Uni-LoRA Adapters (Experts)..."
echo ""

OUTPUT_DIR_S1="${OUTPUT_BASE_DIR}/stage1"

# Launch with deepspeed
/data2/guanbingtao/miniforge3/envs/instruction_tuning/bin/deepspeed --master_port=29505 --include localhost:${CUDA_VISIBLE_DEVICES} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR_S1} \
    --deepspeed ${DS_CONFIG} \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_vector_bank ${LEARNING_RATE_VECTOR_BANK} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm ${MAX_GRAD_NORM} \
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
    --group_by_length \
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

/data2/guanbingtao/miniforge3/envs/instruction_tuning/bin/deepspeed --master_port=29505 --include localhost:${CUDA_VISIBLE_DEVICES} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR_S2} \
    --deepspeed ${DS_CONFIG} \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_vector_bank ${LEARNING_RATE_VECTOR_BANK} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm ${MAX_GRAD_NORM} \
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
    --group_by_length \
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
