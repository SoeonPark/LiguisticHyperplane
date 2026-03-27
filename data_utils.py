"""
data_utils.py

Load HotpotQA data and extract if hallucination is present, and if so, what type of hallucination it is.
"""
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import config
from typing import List, Dict, Tuple, Optional

def build_prompt_no_context(question: str) -> str:
    return f"Question: {question}\nAnswer:"

def build_prompt_with_context(question: str, context: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"

def build_context_string(supporting_facts: List[Tuple[str, str]], paragraphs: Dict[str, str]) -> str:
    context = ""
    for title, sent_id in supporting_facts:
        context += paragraphs[title] + " "
    return context.strip()

# Extract Answer and Supporting Facts from HotpotQA
def extract_answer(generated_answer: str, prompt: str) -> str:
    # Remove the prompt from the generated answer
    if generated_answer.startswith(prompt):
        return generated_answer[len(prompt):].strip()
    return generated_answer.strip()

def normalize_answer(answer: str) -> str:
    return answer.lower().strip()

def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

# Hallucination Types Classifier
def classify_case(correct_w_context: bool, correct_wo_context: bool, 
                           ans_w_context: str, ans_wo_context: str) -> int:
    """
    Returns:
        - Case 1: Answer correct with context, incorrect without context (hallucination type 1)
        - Case 2: Answer incorrect with context, correct without context (hallucination type 2)
        - Case 3: Answer incorrect in both cases with the same answer (hallucination type 3)
        - Case 4: Answer incorrect in both cases with different answers (hallucination type 4)
    """
    
def run_case_filtering(model:str, tokenizer:str, max_samples: Optional[int] = None) -> Dict[str, List[Dict[str, str]]]:
    print("Loading HotpotQA...")
    dataset = load_dataset("hotpotqa/hotpot_qa", split="train") # 여기이거 없음 로컬 -> 이동으로 해야댐
    
    results = {"case_1": [], "case_2": [], "case_3": [], "case_4": []}
    
def generate_answer(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_answer(generated_text, prompt)

def load_model_and_tokenizer(model_name: str):
    print(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    ).to(config.DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer