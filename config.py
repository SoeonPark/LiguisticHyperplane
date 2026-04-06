"""
config.py

Default values only. All settings are overridable via CLI in main.py.
Do NOT hardcode experiment-specific values here — use run.sh instead.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_SPLIT     = "validation"
MAX_SAMPLES    = 2000
ANSWER_TYPES   = ["yes", "no", "entity"]
MAX_NEW_TOKENS = 20

# ── Probe ─────────────────────────────────────────────────────────────────────
PROBE_TEST_SIZE = 0.2
PROBE_MAX_ITER  = 1000
RANDOM_SEED     = 42

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"