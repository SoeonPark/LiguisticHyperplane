"""
Local defaults for RealTimeHalluDetection.

All values are intentionally small wrappers around CLI-overridable behavior in
main.py/custom_cad.py. Keep experiment-specific settings in commands or scripts.
"""

MODEL_NAME = "huggyllama/llama-7b"

MAX_NEW_TOKENS = 100
PROBE_TEST_SIZE = 0.2
PROBE_MAX_ITER = 1000
RANDOM_SEED = 42

DEVICE = "cuda"

OUTPUT_DIR = "outputs/realtime"
HIDDEN_STATE_DIR = "outputs/realtime/hidden_states"
PROBE_RESULT_DIR = "outputs/realtime/probe_results"
PROBE_MODEL_DIR = "outputs/realtime/probe_models"
