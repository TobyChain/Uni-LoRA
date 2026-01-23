#!/bin/bash
# Standard LoRA Training for Qwen MoE with 4-bit Quantization (QLoRA)
# Uses PEFT library with:
# - Flash Attention 2 (memory-efficient attention)
# - Gradient Checkpointing (reduced memory)
# - Fused AdamW (improved training speed)
# - 4-bit NF4 Quantization (QLoRA)

source /root/miniforge3/etc/profile.d/conda.sh
conda activate instruction_tuning
set -euo pipefail

PYTHON=/root/miniforge3/envs/instruction_tuning/bin/python
MODEL_NAME="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
DATASET="/root/autodl-tmp/data/format/qa_alpaca_dedup_exact.jsonl"
OUTPUT_DIR="./output/standard_lora_4bit"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "=== Standard LoRA Training (4-bit QLoRA) with Optimizations ==="
echo "Optimizations: Flash Attention 2, Gradient Checkpointing, Fused AdamW, 4-bit NF4"
echo "Output: $OUTPUT_DIR"

mkdir -p ${OUTPUT_DIR}

${PYTHON} train_standard_lora.py \
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
    --learning_rate 2e-4 \
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
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --use_flash_attention_2 True \
    --use_fused_adamw True \
    --bits 4 \
    --double_quant True \
    --quant_type nf4 \
    --gradient_checkpointing \
    --save_total_limit 2 \
    --logging_steps 10 \
    --report_to tensorboard \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "✅ Standard LoRA (4-bit QLoRA) Training Completed!"
