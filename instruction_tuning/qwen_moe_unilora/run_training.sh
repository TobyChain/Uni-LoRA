#!/bin/bash

set -euo pipefail

# 便捷训练启动脚本
# 从 qwen_moe_unilora 根目录运行
#
# 使用方法:
#   前台运行: bash run_training.sh
#   后台运行: bash run_training.sh &
#   或者: nohup bash run_training.sh > training.log 2>&1 &

cd "$(dirname "$0")/training"
bash run_unilora_pipeline.sh

