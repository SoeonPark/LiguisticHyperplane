#!/bin/bash
# run_0.sh
# LLaMA-3.1-8B Base — Main experiment
# GPU: 0
#
# Usage:
#   bash run_0.sh               # full pipeline
#   bash run_0.sh probe         # probe only (skip data/extract if cached)
#   bash run_0.sh data          # data filtering only

# nohup bash run_3.sh > llama3_firstToken_balanced.log 2>&1 &
# nohup bash run_3.sh probe > llama3_probe.log 2>&1 &
# nohup bash run_3.sh visualize > llama3_visualize.log 2>&1 &

export CUDA_VISIBLE_DEVICES="3"

# ── Experiment settings ────────────────────────────────────────────────────────
MODEL="meta-llama/Meta-Llama-3.1-8B"
GPU="3"
PHASE=${1:-all}

SUBSET="both"           # fullwiki + distractor, merged
DATA_SPLIT="validation" # validation 전체 (7,405개 × 2 subset = ~14,810 raw)
MAX_SAMPLES=""          # 비워두면 split 전체 사용 (None)
STRATEGY="first"          # first + mean + last 모두 추출/분석
BALANCED="--balanced"             # "--balanced" 로 바꾸면 class_weight='balanced' 활성화
TAG=$([ -n "$BALANCED" ] && echo "balanced" || echo "") # default: no suffix, balanced: "balanced" suffix

# ── Display ────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  LLaMA-3.1-8B  |  Linguistic Hyperplane"
echo "  Model    : $MODEL"
echo "  Subset   : $SUBSET"
echo "  Split    : $DATA_SPLIT  (max_samples=${MAX_SAMPLES:-all})"
echo "  Strategy : $STRATEGY"
echo "  Phase    : $PHASE"
echo "  GPU      : $GPU"
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

if [ -n "$MAX_SAMPLES" ]; then
    ARGS="$ARGS --max_samples $MAX_SAMPLES"
fi
if [ -n "$TAG" ]; then
    ARGS="$ARGS --tag $TAG"
fi
if [ -n "$BALANCED" ]; then
    ARGS="$ARGS $BALANCED"
fi

python main.py $ARGS

echo ""
echo "========================================"
echo "  Done: LLaMA / $PHASE"
echo "========================================"