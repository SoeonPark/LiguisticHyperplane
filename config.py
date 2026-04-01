import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"   # Base model (not instruct)
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# MODEL_NAME = "microsoft/phi-2"               # For quick pipeline test

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_SPLIT    = "validation"
MAX_SAMPLES   = 2000
ANSWER_TYPES  = ["yes", "no", "entity"]        # Exclude long-form answers
BATCH_SIZE    = 8
MAX_NEW_TOKENS = 20                            # Max tokens to generate per answer

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "outputs"
LOG_DIR          = os.path.join(OUTPUT_DIR, "logs")
CASE_DATA_PATH   = os.path.join(OUTPUT_DIR, "cases.json")
HIDDEN_STATE_DIR = os.path.join(OUTPUT_DIR, "hidden_states")
PROBE_RESULT_DIR = os.path.join(OUTPUT_DIR, "probe_results")
FIGURE_DIR       = os.path.join(OUTPUT_DIR, "figures")

# ── Hidden State Extraction ───────────────────────────────────────────────────
# Phase 2 default: "first"
# Phase 4 comparison: ["first", "mean", "last"]
TOKEN_POSITIONS = ["first"]

# ── Linear Probe ──────────────────────────────────────────────────────────────
PROBE_TEST_SIZE = 0.2
PROBE_MAX_ITER  = 1000
RANDOM_SEED     = 42

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"