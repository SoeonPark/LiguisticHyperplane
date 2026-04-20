import re
import os
import json
from typing import Dict, List, Optional, Tuple

import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
import pandas as pd

import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria]

# Prompt Builders

def build_prompt_wo_context(question: str) -> str:
    return f"Question: {question}\nAnswer:"

def build_prompt_w_context(question: str, context: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"

# Loading Data

# DoLa set
def load_truthfulqa(subset: str, split: str, max_samples: Optional[int]) -> Tuple[list, str]:
    """
    Load TruthfulQA dataset (primarily for DoLa evaluation).
    
    subset: 'multiple_choice' | 'generation'
    split:  'validation' (TruthfulQA typically only has validation)
    """
    # 기본값이 지정되지 않았다면 multiple_choice를 디폴트로 사용
    subset = subset if subset in ["multiple_choice", "generation"] else "multiple_choice"
    
    print(f"  Loading TruthfulQA [{subset} / {split}]...")
    dataset = load_dataset("truthful_qa", subset, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"  -> {len(dataset)} samples")
    
    return dataset, subset

# CAD set
def load_nq_swap(subset: str, split: str, max_samples: Optional[int]) -> Tuple[list, str]:
    """
    Load NQ-Swap dataset (primarily for Context-Aware Decoding evaluation).
    
    subset: HF repo name (e.g., 'lucasmccabe-lmi/nq-swap') or 'local'
    split:  'validation' | 'test'
    """
    print(f"  Loading NQ-Swap [{subset} / {split}]...")
    
    try:
        if subset == "local":
            data_files = {split: f"data/nq_swap_{split}.json"}
            dataset = load_dataset("json", data_files=data_files, split=split)
        else:
            repo_name = subset if "/" in subset else "lucasmccabe-lmi/nq-swap" 
            dataset = load_dataset(repo_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"  -> {len(dataset)} samples")
    
    return dataset, "nq_swap"
