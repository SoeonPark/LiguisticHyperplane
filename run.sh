#!/bin/bash
# run.sh
# Convenience wrapper for the Linguistic Hyperplane pipeline.
#
# Usage:
#   bash run.sh [phase] [strategy]
#
# Examples:
#   bash run.sh all first          # full pipeline, first-token strategy
#   bash run.sh data               # data filtering only
#   bash run.sh probe mean         # probe training with mean pooling
#   bash run.sh token_pos          # compare first/mean/last
set -e

PHASE=${1:-all}
STRATEGY=${2:-first}
GPU=${3:-0} 

export CUDA_VISIBLE_DEVICES=$GPU

echo "========================================"
echo "  Linguistic Hyperplane Pipeline"
echo "  Phase    : $PHASE"
echo "  Strategy : $STRATEGY"
echo "  GPU      : $GPU"
echo "========================================"

echo ""
echo "[Check] Python environment..."
python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers={transformers.__version__}')"
python -c "import datasets; print(f'  datasets={datasets.__version__}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[GPU Status]"
    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
fi

echo ""

python main.py --phase "$PHASE" --strategy "$STRATEGY"

echo ""
echo "========================================"
echo "  Done: $PHASE"
echo "========================================"