import os

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"  
# MODEL_NAME = "microsoft/phi-2"  

DATA_SPLIT = "validation"
MAX_SAMPLES = 2000
ANSWER_TYPES = ["yes", "no", "entity"]  
BATCH_SIZE = 8

OUTPUT_DIR = "outputs"
CASE_DATA_PATH = os.path.join(OUTPUT_DIR, "cases.json")
HIDDEN_STATE_DIR = os.path.join(OUTPUT_DIR, "hidden_states")
PROBE_RESULT_DIR = os.path.join(OUTPUT_DIR, "probe_results")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

TOKEN_POSITIONS = ["first"]   # "first" | "mean" | "last" | "all"

PROBE_TEST_SIZE = 0.2
PROBE_MAX_ITER = 1000
RANDOM_SEED = 42

DEVICE= "cuda"