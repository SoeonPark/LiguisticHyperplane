"""
data_utils.py

Load HotpotQA and filter into Case 1 (non-hallucination) and Case 3 (hallucination).

Supports two HotpotQA subset configurations:
  fullwiki   : gold context from supporting facts only (harder for model)
  distractor : gold context mixed with 10 distractor paragraphs
  both       : run fullwiki and distractor, merge results (deduped by question+subset)

Case definitions:
  Case 1: Correct w/ context, Wrong w/o context  -> label 0 (non-hallucination)
  Case 2: Wrong w/ context, Correct w/o context  -> excluded
  Case 3: Wrong in both, same answer              -> label 1 (hallucination)
  Case 4: Wrong in both, different answers        -> excluded
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import re, string

import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_prompt_no_context(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_prompt_with_context(question: str, context: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def build_context_string(
    supporting_facts_titles: List[str],
    paragraphs: List[Tuple[str, List[str]]],
) -> str:
    """Build a context string from HotpotQA supporting facts titles."""
    para_dict = {title: "".join(sents) for title, sents in paragraphs}
    context_parts = []
    seen = set()
    for title in supporting_facts_titles:
        if title not in seen and title in para_dict:
            context_parts.append(para_dict[title])
            seen.add(title)
    return " ".join(context_parts)


# ── Answer Utilities ──────────────────────────────────────────────────────────

# def extract_answer(generated_text: str, prompt: str) -> str:
#     if generated_text.startswith(prompt):
#         answer = generated_text[len(prompt):].strip()
#     else:
#         answer = generated_text.strip()
#     return answer.split("\n")[0].strip()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(text.split())



def is_correct(pred: str, gold: str) -> bool:
    """
        If Comparison type, Check for exact match. 
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # For the Comparison Type
    if gold_norm in ('yes', 'no'):
        first = pred_norm.split()[0] if pred_norm.split() else ''
        return first == gold_norm

    # For the Entity Type
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if not pred_tokens or not gold_tokens:
        return False
    common = pred_tokens & gold_tokens
    if not common:
        return False
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= 0.5

# ── Case Classifier ───────────────────────────────────────────────────────────

def classify_case(
    correct_w_context: bool,
    correct_wo_context: bool,
    ans_w_context: str,
    ans_wo_context: str,
) -> int:
    if correct_w_context and correct_wo_context:
        return 0
    elif correct_w_context and not correct_wo_context:
        return 1   # Non-hallucination
    elif not correct_w_context and correct_wo_context:
        return 2   # Excluded
    else:
        if normalize_answer(ans_w_context) == normalize_answer(ans_wo_context):
            return 3   # Hallucination
        else:
            return 4   # Excluded


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, prompt: str) -> str:
    """Firstly generate the answer using the model, then extract the answer part from the generated text."""
    # Generate
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    
    # breakpoint()
    """
    (Pdb) print(tokenizer.all_special_ids)
    [128000, 128001]
    (Pdb) prompt
    'Question: Were Scott Derrickson and Ed Wood of the same nationality?\nAnswer:'
    (Pdb) len(inputs)
    2
    (Pdb) inputs["input_ids"].shape
    torch.Size([1, 17])
    (Pdb) len(inputs["input_ids"])
    1
    (Pdb) input_len
    17
    (Pdb) inputs
    {'input_ids': tensor([[128000,  14924,     25,  40070,  10016,  73189,    942,    323,   3279,
            12404,    315,    279,   1890,  59343,   5380,  16533,     25]],
        device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}
    """
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Extract answer part
    new_ids = output_ids[0, input_len:] # Tokens that newly generated
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    # answer_raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    # breakpoint()
    """
    (Pdb) answer
    ' No. Scott Derrickson is an American film director, screenwriter, and producer. Ed Wood was'
    """
    return answer.split("\n")[0].strip()

# ── Dataset Loader ────────────────────────────────────────────────────────────

def load_hotpotqa(subset: str, split: str, max_samples: Optional[int]) -> list:
    """
    Load a single HotpotQA subset/split and optionally subsample.

    subset: 'fullwiki' | 'distractor'
    split:  'train' | 'validation'
    """
    print(f"  Loading HotpotQA [{subset} / {split}]...")
    dataset = load_dataset("hotpotqa/hotpot_qa", subset, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"  -> {len(dataset)} samples")
    
    return dataset, subset


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def _filter_single_dataset(
    dataset,
    subset_name: str,
    model,
    tokenizer,
    answer_types: List[str],
) -> List[Dict]:
    """Run case filtering on a single HotpotQA dataset object."""
    results = []
    model.eval()

    for item in tqdm(dataset, desc=f"Filtering [{subset_name}]"):
        question      = item["question"]
        gold_answer   = item["answer"]
        question_type = item["type"]

        if gold_answer.lower() in ["yes", "no"]:
            answer_type = gold_answer.lower()
        else:
            answer_type = "entity"

        if answer_type not in answer_types:
            continue

        supporting_facts_titles = [sf[0] for sf in item["supporting_facts"]]
        paragraphs = list(
            zip(item["context"]["title"], item["context"]["sentences"])
        )
        context = build_context_string(supporting_facts_titles, paragraphs)

        prompt_wo_context = build_prompt_no_context(question)
        prompt_w_context  = build_prompt_with_context(question, context)

        ans_wo_context = generate_answer(model, tokenizer, prompt_wo_context)
        ans_w_context  = generate_answer(model, tokenizer, prompt_w_context)

        correct_wo_context = is_correct(ans_wo_context, gold_answer)
        correct_w_context  = is_correct(ans_w_context,  gold_answer)

        case = classify_case(
            correct_w_context,
            correct_wo_context,
            ans_w_context,
            ans_wo_context,
        )

        # label = 0 if case == 1 else 1
        label = 0 if case == 1 else (1 if case == 3 else None)

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "answer_type": answer_type,
            "question_type": question_type,
            "subset": subset_name,
            "context": context,
            "ans_wo_context": ans_wo_context,
            "ans_w_context": ans_w_context,
            "case": case,
            "label": label,
            "prompt_wo_context": prompt_wo_context,
            "prompt_w_context": prompt_w_context,
        })

    return results


def run_case_filtering(
    model,
    tokenizer,
    max_samples: Optional[int] = None,
    answer_types: Optional[List[str]] = None,
    subset: str = "both",          # "fullwiki" | "distractor" | "both"
    data_split: str = "validation",
) -> List[Dict]:
    """
    Run inference on HotpotQA and collect Case 1 / Case 3 samples.

    subset="both"  -> runs fullwiki and distractor, merges all results.
    max_samples    -> per-subset cap (so "both" can yield up to 2×max_samples raw).
    data_split     -> "train" | "validation"
    """
    allowed_types = answer_types if answer_types is not None else config.ANSWER_TYPES

    subsets_to_run = (
        ["fullwiki", "distractor"] if subset == "both" else [subset]
    )

    all_results: List[Dict] = []

    for ss in subsets_to_run:
        print(f"\n{'─'*50}")
        print(f"  Subset : {ss}  |  Split : {data_split}")
        print(f"{'─'*50}")
        dataset, _ = load_hotpotqa(ss, data_split, max_samples)
        results = _filter_single_dataset(dataset, ss, model, tokenizer, allowed_types)
        print(f"[DBG] Example result: {results[0] if results else 'No results'}")
        # breakpoint()
        all_results.extend(results)

        n1 = sum(1 for r in results if r["case"] == 1)
        n3 = sum(1 for r in results if r["case"] == 3)
        print(f"\n  [{ss}] processed={len(dataset)}  "
              f"case1(non-hall)={n1}  case3(hall)={n3}  kept={len(results)}")
        
    total = len(all_results)
    for c in range(5):
        n = sum(1 for r in all_results if r["case"] == c)
        print(f"  Case {c}: {n:>5}  ({n/total*100:.1f}%)") 

    # ── Grand total stats ──────────────────────────────────────────────────
    n1_total = sum(1 for r in all_results if r["case"] == 1)
    n3_total = sum(1 for r in all_results if r["case"] == 3)
    print(f"\n{'='*50}")
    print(f"Case filtering complete (subset={subset}, split={data_split})")
    print(f"  Total kept      : {len(all_results)}")
    print(f"  Case 1 (label 0, non-hallucination) : {n1_total}")
    print(f"  Case 3 (label 1, hallucination)     : {n3_total}")
    print(f"{'='*50}")

    return all_results


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_cases(cases: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} cases -> {path}")


def load_cases(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: Optional[str] = None):
    name = model_name or config.MODEL_NAME
    print(f"Loading model: {name}")

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer