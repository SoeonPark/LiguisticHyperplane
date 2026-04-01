import argparse
import os
import json
import torch

import data_utils
import config
import extract_hidden_state as ehs
import linear_probe as lp
import layer_analysis as la

# Data Preparation
def prepare_data(model, tokenizer):
    
def extract_hidden_states():
    
def train_probes():
    
def analyze_layers():
    
def visualize_results():
    
def compare_by_token_position():
    
def main():
    parser = argparse.ArgumentParser(description="Linguistic Hyperplane Analysis")
    parser.add_argument("--phase", type=int, default=1, help="Phase to run (1-4)",
                        choices=["all", "train_probes", "analyze_layers", "visualize_results", "compare_by_token_position"])
    parser.add_argument("--token_strategy", type=str, default="first", help="Token pooling strategy for answer representation",
                        choices=["first", "mean", "last", "all"])
    args = parser.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.HIDDEN_STATE_DIR, exist_ok=True)
    os.makedirs(config.PROBE_RESULT_DIR, exist_ok=True)
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    
    if args.phase in ["all"]:
        prepare_data()
        extract_hidden_states()
        train_probes()
        analyze_layers()
        visualize_results()
        compare_by_token_position()
    elif args.phase == "train_probes":
        train_probes()
    elif args.phase == "analyze_layers":
        analyze_layers()
    elif args.phase == "visualize_results": 
        visualize_results()
    elif args.phase == "compare_by_token_position":
        compare_by_token_position()
    

if __name__ == "__main__":
    main()