#!/bin/bash
# run_0.sh
# LLaMA-3.1-8B Base — Main experiment
# GPU: 0
#
# Usage:
#   bash run_0.sh               # full pipeline, all strategies
#   bash run_0.sh probe         # probe only
#   bash run_0.sh probe first   # probe, first strategy only
#   bash run_0.sh visualize
#   bash run_0.sh data

# nohup bash run_0.sh > llama3.log 2>&1 &
# nohup bash run_0.sh probe > llama3_probe.log 2>&1 &

export CUDA_VISIBLE_DEVICES="0"

# ── Experiment settings ────────────────────────────────────────────────────────
MODEL="meta-llama/Meta-Llama-3.1-8B"
GPU="0"
PHASE=${1:-all}
STRATEGY=${2:-all}      # first | mean | last | all(기본)

SUBSET="both"           # fullwiki + distractor, merged
DATA_SPLIT="validation"
MAX_SAMPLES=""          # 비워두면 split 전체 사용
BALANCED=""             # "--balanced" 로 바꾸면 class_weight='balanced' 활성화
TAG=""                  # 추가 suffix 없음

# ── Display ────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  LLaMA-3.1-8B  |  Linguistic Hyperplane"
echo "  Model        : $MODEL"
echo "  Subset       : $SUBSET"
echo "  Split        : $DATA_SPLIT  (max_samples=${MAX_SAMPLES:-all})"
echo "  Strategy     : $STRATEGY"
echo "  Phase        : $PHASE"
echo "  GPU          : $GPU"
echo "========================================"

echo ""
echo "[Check] Python environment..."
python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers={transformers.__version__}')"
python -c "import datasets;     print(f'  datasets={datasets.__version__}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[GPU Status]"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader
fi

echo ""

# ── Run ────────────────────────────────────────────────────────────────────────
ARGS="--model $MODEL \
      --phase $PHASE \
      --strategy $STRATEGY \
      --subset $SUBSET \
      --data_split $DATA_SPLIT \
      --gpu $GPU"

[ -n "$MAX_SAMPLES" ] && ARGS="$ARGS --max_samples $MAX_SAMPLES"
[ -n "$TAG" ]         && ARGS="$ARGS --tag $TAG"
[ -n "$BALANCED" ]    && ARGS="$ARGS $BALANCED"

python main.py $ARGS

echo ""
echo "========================================"
echo "  Done: LLaMA / $PHASE / $STRATEGY"
echo "========================================"