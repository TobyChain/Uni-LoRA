#!/bin/bash
# Environment check script for Standard LoRA training
# Run this before training to verify all dependencies are correctly configured

echo "=============================================="
echo "       Environment Check Script              "
echo "=============================================="

# Activate conda environment
source /root/miniforge3/etc/profile.d/conda.sh
conda activate instruction_tuning

echo ""
echo "=== 1. Python Environment ==="
echo "Python: $(python --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV"

echo ""
echo "=== 2. GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "=== 3. CUDA Configuration ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
"

echo ""
echo "=== 4. Key Dependencies ==="
python -c "
import importlib
packages = [
    ('transformers', 'transformers'),
    ('peft', 'peft'),
    ('bitsandbytes', 'bitsandbytes'),
    ('accelerate', 'accelerate'),
    ('safetensors', 'safetensors'),
    ('datasets', 'datasets'),
]

for name, pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'{name}: {version} ✓')
    except ImportError:
        print(f'{name}: NOT INSTALLED ✗')
"

echo ""
echo "=== 5. Flash Attention 2 ==="
python -c "
try:
    import flash_attn
    print(f'Flash Attention version: {flash_attn.__version__} ✓')
except ImportError:
    print('Flash Attention: NOT INSTALLED')
    print('Install with: pip install flash-attn --no-build-isolation')
"

echo ""
echo "=== 6. Fused AdamW Support ==="
python -c "
import torch
try:
    params = [torch.nn.Parameter(torch.randn(10, 10, device='cuda'))]
    opt = torch.optim.AdamW(params, lr=1e-4, fused=True)
    print('Fused AdamW: SUPPORTED ✓')
except Exception as e:
    print(f'Fused AdamW: NOT SUPPORTED')
    print(f'Error: {e}')
"

echo ""
echo "=== 7. Model Path ==="
MODEL_PATH="/root/autodl-tmp/model/Qwen/Qwen1.5-MoE-A2.7B-Chat"
if [ -d "$MODEL_PATH" ]; then
    echo "Model path: $MODEL_PATH ✓"
    ls -lh "$MODEL_PATH" | head -5
else
    echo "Model path: $MODEL_PATH ✗ NOT FOUND"
fi

echo ""
echo "=== 8. Dataset Path ==="
DATA_PATH="/root/autodl-tmp/data/format/qa_alpaca_dedup_exact.jsonl"
if [ -f "$DATA_PATH" ]; then
    echo "Dataset path: $DATA_PATH ✓"
    echo "File size: $(du -h $DATA_PATH | cut -f1)"
    echo "Sample (first line):"
    head -1 "$DATA_PATH" | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(f'  Keys: {list(d.keys())}')"
else
    echo "Dataset path: $DATA_PATH ✗ NOT FOUND"
    echo "Searching for available datasets..."
    find /root/autodl-tmp -name "*.jsonl" -type f 2>/dev/null | head -5
fi

echo ""
echo "=============================================="
echo "       Check Complete                        "
echo "=============================================="
