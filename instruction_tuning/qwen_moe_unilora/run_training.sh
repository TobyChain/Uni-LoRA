#!/bin/bash

# 便捷训练启动脚本
# 从 qwen_moe_unilora 根目录运行

cd "$(dirname "$0")/training"
bash finetune_qwem1.5_moe_A2.7_unilora_1229.sh

