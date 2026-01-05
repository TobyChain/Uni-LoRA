#!/bin/bash

# run_unilora_pipeline.sh
# Unified pipeline for Uni-LoRA + MoE training (Stage 1 + Stage 2)

set -euo pipefail

# === Configuration ===
MODEL_NAME=${1:-"Qwen/Qwen1.5-MoE-A2.7B-Chat"}
DATASET=${2:-"alpaca-clean"}
OUTPUT_BASE_DIR=${3:-"./output/qwen_moe_unilora_pipeline"}

RANK=8
ALPHA=16.0
USE_MATRIX_MODE=True
RESERVE_MEMORY_MB=512
USE_QUANTIZATION=False

# Training Hyperparameters
# 双卡 bf16 半精度训练 + DeepSpeed ZeRO-2
NUM_GPUS=2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,5"}
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=24
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=3
SAVE_STEPS=500
EVAL_STEPS=500
MAX_STEPS=-1 # -1 for full epochs

# DeepSpeed Config
DS_CONFIG="../configs/ds_config.json"
if [ ! -f "$DS_CONFIG" ]; then
    echo "Warning: DeepSpeed config not found at $DS_CONFIG. Checking current directory..."
    if [ -f "ds_config.json" ]; then
        DS_CONFIG="ds_config.json"
    else
        # Try finding it relative to the script location
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        if [ -f "$SCRIPT_DIR/../configs/ds_config.json" ]; then
             DS_CONFIG="$SCRIPT_DIR/../configs/ds_config.json"
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
echo "Starting Uni-LoRA Pipeline"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_BASE_DIR"
echo "=========================================="

mkdir -p ${OUTPUT_BASE_DIR}

# === Stage 1: Adapter Training (Router Frozen) ===
echo ""
echo ">>> [STAGE 1] Training Uni-LoRA Adapters..."
echo ""

OUTPUT_DIR_S1="${OUTPUT_BASE_DIR}/stage1"

# Use deepspeed absolute path to avoid PATH issues
/data2/guanbingtao/miniforge3/envs/instruction_tuning/bin/deepspeed --master_port=29505 --include localhost:${CUDA_VISIBLE_DEVICES} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
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
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_steps 0.03 \
    --save_strategy steps \
    --eval_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --rank ${RANK} \
    --alpha ${ALPHA} \
    --training_stage 1 \
    $([ "${USE_MATRIX_MODE}" = "True" ] && echo "--use_matrix_mode") \
    --reserve_memory_mb ${RESERVE_MEMORY_MB} \
    --gradient_checkpointing \
    --group_by_length \
    --save_total_limit 2 \
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

# Note: The python script automatically finds the best checkpoint from stage1_output_dir 
# if we pass the correct structure, but our script logic in main() expects 
# output_dir to be the CURRENT stage output, and it looks for ../stage1.
# So if we make OUTPUT_DIR_S2 = OUTPUT_BASE_DIR/stage2, then ../stage1 is OUTPUT_BASE_DIR/stage1.
# This matches the logic in train_unilora_moe.py:
# stage1_output_dir = os.path.join(training_args.output_dir, "..", "stage1")

# Use deepspeed absolute path
/data2/guanbingtao/miniforge3/envs/instruction_tuning/bin/deepspeed --master_port=29505 --include localhost:${CUDA_VISIBLE_DEVICES} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
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
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_steps 0.03 \
    --save_strategy steps \
    --eval_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --rank ${RANK} \
    --alpha ${ALPHA} \
    --training_stage 2 \
    $([ "${USE_MATRIX_MODE}" = "True" ] && echo "--use_matrix_mode") \
    --reserve_memory_mb ${RESERVE_MEMORY_MB} \
    --gradient_checkpointing \
    --group_by_length \
    --save_total_limit 2 \
    2>&1 | tee ${OUTPUT_BASE_DIR}/stage2_training.log

ST2_STATUS=${PIPESTATUS[0]}
if [ ${ST2_STATUS} -ne 0 ]; then
    echo "❌ Stage 2 Failed! Exit Code: ${ST2_STATUS}"
    exit ${ST2_STATUS}
fi

echo "✅ Stage 2 Completed Successfully."
echo "🎉 Full Uni-LoRA Pipeline Finished!"
