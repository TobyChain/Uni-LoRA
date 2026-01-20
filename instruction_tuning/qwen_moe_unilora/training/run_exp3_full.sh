#!/bin/bash
# Experiment 3: Full Combined (MoE + Router + Attention)
# Train everything together

source /root/miniconda3/etc/profile.d/conda.sh
conda activate instruction_tuning
set -euo pipefail

PYTHON=/root/miniconda3/envs/instruction_tuning/bin/python
MODEL_NAME="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
DATASET="/root/autodl-tmp/data/fomat/qa_alpaca_dedup_exact.jsonl"
OUTPUT_DIR="./output/exp3_full_combined"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "=== Experiment 3: Full Combined ==="
echo "Output: $OUTPUT_DIR"

mkdir -p ${OUTPUT_DIR}

${PYTHON} train_unilora_moe.py \
    --model_name_or_path ${MODEL_NAME} \
    --trust_remote_code \
    --dataset ${DATASET} \
    --dataset_format alpaca \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --do_eval \
    --bf16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --learning_rate_vector_bank 1e-3 \
    --learning_rate_router 1e-5 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --eval_strategy steps \
    --load_best_model_at_end \
    --lora_r 64 \
    --lora_alpha 16.0 \
    --training_stage 0 \
    --shared_vector_rank 8 \
    --train_attention True \
    --attn_shared_vector_rank 8 \
    --gradient_checkpointing \
    --save_total_limit 2 \
    --logging_steps 10 \
    --report_to swanlab \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "✅ Experiment 3 Completed!"
