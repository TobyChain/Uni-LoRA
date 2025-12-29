#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 nohup python -u qlora_unilora.py \
    --model_name_or_path /data4/guanbingtao/model/meta-llama/llama-2-7b-hf \
    --output_dir /data4/guanbingtao/Uni-LoRA/instruction_tuning/output/llama2_7b_unilora_local \
    --logging_steps 20 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy no \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 4 \
    --num_vectors 2048 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --gradient_checkpointing \
    --dataset alpaca-clean \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 0 \
    --learning_rate_vector_bank 8e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0 \
    > /data4/guanbingtao/Uni-LoRA/instruction_tuning/finetune_llama2_7b_unilora_20251221.log 2>&1 &