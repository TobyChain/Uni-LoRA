#!/usr/bin/env python3
"""
修复现有checkpoint - 添加缺失的config.json文件
"""
import os
import sys
from transformers import AutoConfig

# 设置路径
model_path = "/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
checkpoint_path = "./output/exp2_attn_only/checkpoint-1000"

print(f"Loading config from: {model_path}")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print(f"Saving config to: {checkpoint_path}")
config.save_pretrained(checkpoint_path)

print("✓ Checkpoint fixed! config.json has been added.")
print(f"\nCheckpoint directory now contains:")
os.system(f"ls -la {checkpoint_path}")
