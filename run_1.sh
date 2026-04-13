#!/bin/bash
# run_0.sh
# LLaMA-3.1-8B Base — Main experiment
# GPU: 0
#
# Usage:
#   bash run_0.sh               # full pipeline
#   bash run_0.sh probe         # probe only (skip data/extract if cached)
#   bash run_0.sh data          # data filtering only

# nohup bash run_1.sh > mistral_0409.log 2>&1 &
# nohup bash run_1.sh > llama2_13b.log 2>&1 &
# nohup bash run_1.sh probe > llama2_13b_probe.log 2>&1 &
# nohup bash run_1.sh visualize > llama2_13b_visualize.log 2>&1 &

# ── Experiment settings ────────────────────────────────────────────────────────
MODEL="mistralai/Mistral-7B-v0.1"  #"meta-llama/Meta-Llama-3.1-8B"
GPU="1"
PHASE=${1:-all}
if [ "$#" -ge 2 ]; then
    STRATEGIES=("${@:2}")   # 예: bash run_1.sh probe mean last
else
    STRATEGIES=("all")
fi

SUBSET="distractor"           # fullwiki + distractor, merged
DATA_SPLIT="validation"
MAX_SAMPLES=""          # 비워두면 split 전체 사용
BALANCED=""             # "--balanced" 로 바꾸면 class_weight='balanced' 활성화
TAG_SUFFIX=""

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

run_main() {
    local RUN_PHASE="$1"
    local RUN_STRATEGY="$2"
    local TAG="${RUN_STRATEGY}_${TAG_SUFFIX}"

    echo "========================================"
    echo "  $MODEL  |  Linguistic Hyperplane"
    echo "  Model        : $MODEL"
    echo "  Subset       : $SUBSET"
    echo "  Split        : $DATA_SPLIT  (max_samples=${MAX_SAMPLES:-all})"
    echo "  Strategy     : $RUN_STRATEGY"
    echo "  Phase        : $RUN_PHASE"
    echo "  GPU          : $GPU"
    echo "========================================"

    ARGS="--model $MODEL \
          --phase $RUN_PHASE \
          --strategy $RUN_STRATEGY \
          --subset $SUBSET \
          --data_split $DATA_SPLIT \
          --gpu $GPU"

    [ -n "$MAX_SAMPLES" ] && ARGS="$ARGS --max_samples $MAX_SAMPLES"
    [ -n "$TAG" ]         && ARGS="$ARGS --tag $TAG"
    [ -n "$BALANCED" ]    && ARGS="$ARGS $BALANCED"

    python main.py $ARGS

    echo ""
    echo "========================================"
    echo "  Done: $MODEL / $RUN_PHASE / $RUN_STRATEGY"
    echo "========================================"
}

if [ "$PHASE" = "all" ]; then
    echo "[Info] PHASE=all -> running:"
    echo "       data once, then full downstream analysis for each strategy."
    echo "       token_pos is skipped here because it always runs first/mean/last."
    echo ""

    run_main "data" "${STRATEGIES[0]}"
    for STRATEGY in "${STRATEGIES[@]}"; do
        run_main "extract" "$STRATEGY"
        run_main "probe" "$STRATEGY"
        run_main "visualize" "$STRATEGY"
        run_main "probe_direction" "$STRATEGY"
        run_main "pca" "$STRATEGY"
        run_main "cka" "$STRATEGY"
    done
    run_main "attention" "${STRATEGIES[0]}"
elif [ "$PHASE" = "data" ] || [ "$PHASE" = "token_pos" ] || [ "$PHASE" = "attention" ]; then
    run_main "$PHASE" "${STRATEGIES[0]}"
else
    for STRATEGY in "${STRATEGIES[@]}"; do
        run_main "$PHASE" "$STRATEGY"
    done
fi
