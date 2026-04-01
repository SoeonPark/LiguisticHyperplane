"""
data_utils.py

Load HotpotQA and filter into Case 1 (non-hallucination) and Case 3 (hallucination)
by running inference under two conditions:
  Condition A: Question only          (w/o context)
  Condition B: Context + Question     (w/  context)

Case definitions:
  Case 1: Correct w/ context, Wrong w/o context  → label 0 (non-hallucination)
  Case 2: Wrong w/ context, Correct w/o context  → excluded
  Case 3: Wrong in both, same answer              → label 1 (hallucination)
  Case 4: Wrong in both, different answers        → excluded
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
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
    """Build a context string from HotpotQA supporting facts."""
    para_dict = {title: "".join(sents) for title, sents in paragraphs}
    context_parts = []
    seen = set()
    for title in supporting_facts_titles:
        if title not in seen and title in para_dict:
            context_parts.append(para_dict[title])
            seen.add(title)
    return " ".join(context_parts)


# ── Answer Utilities ──────────────────────────────────────────────────────────

def extract_answer(generated_text: str, prompt: str) -> str:
    """Strip the prompt prefix and return only the generated answer."""
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        answer = generated_text.strip()
    # Keep only the first line (greedy decoding typically gives one answer)
    return answer.split("\n")[0].strip()


def normalize_answer(text: str) -> str:
    return text.lower().strip()


def is_correct(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return pred_norm == gold_norm or gold_norm in pred_norm


# ── Case Classifier ───────────────────────────────────────────────────────────

def classify_case(
    correct_w_context: bool,
    correct_wo_context: bool,
    ans_w_context: str,
    ans_wo_context: str,
) -> int:
    """
    Returns:
        1: Correct w/ context, Wrong w/o context → Non-hallucination (label 0)
        2: Wrong w/ context, Correct w/o context → Excluded
        3: Wrong in both with same answer         → Hallucination (label 1)
        4: Wrong in both with different answers   → Excluded
        0: Both correct                           → Excluded (cannot distinguish)
    """
    if correct_w_context and correct_wo_context:
        return 0   # Both correct — model already knew, cannot distinguish
    elif correct_w_context and not correct_wo_context:
        return 1   # Context helped → non-hallucination
    elif not correct_w_context and correct_wo_context:
        return 2   # Context hurt  → excluded
    else:
        # Both wrong
        if normalize_answer(ans_w_context) == normalize_answer(ans_wo_context):
            return 3   # Consistent wrong answer → hallucination
        else:
            return 4   # Inconsistent wrong answers → excluded


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, prompt: str) -> str:
    """Generate a short answer with greedy decoding."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_answer(generated_text, prompt)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_case_filtering(
    model,
    tokenizer,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Run inference on HotpotQA and collect Case 1 and Case 3 samples.

    Returns:
        List of dicts containing question, answer, context, predictions,
        case number, and binary label (0 = non-hallucination, 1 = hallucination).
    """
    print("Loading HotpotQA (fullwiki)...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=config.DATA_SPLIT)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    results = []
    model.eval()

    for item in tqdm(dataset, desc="Filtering cases"):
        question      = item["question"]
        gold_answer   = item["answer"]
        question_type = item["type"]   # "bridge" or "comparison"

        # ── Answer type filtering ──────────────────────────────────────────
        # For comparison questions the answer is typically "yes" / "no".
        # For bridge questions the answer is usually a named entity.
        if gold_answer.lower() in ["yes", "no"]:
            answer_type = gold_answer.lower()
        else:
            answer_type = "entity"

        if answer_type not in config.ANSWER_TYPES:
            continue

        # ── Context construction ───────────────────────────────────────────
        supporting_facts_titles = [sf[0] for sf in item["supporting_facts"]]
        paragraphs = list(
            zip(item["context"]["title"], item["context"]["sentences"])
        )
        context = build_context_string(supporting_facts_titles, paragraphs)

        prompt_wo_context = build_prompt_no_context(question)
        prompt_w_context  = build_prompt_with_context(question, context)

        # ── Inference ──────────────────────────────────────────────────────
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

        # We only use Case 1 (non-hallucination) and Case 3 (hallucination)
        if case not in [1, 3]:
            continue

        label = 0 if case == 1 else 1

        results.append({
            "question":          question,
            "gold_answer":       gold_answer,
            "answer_type":       answer_type,
            "question_type":     question_type,
            "context":           context,
            "ans_wo_context":    ans_wo_context,
            "ans_w_context":     ans_w_context,
            "case":              case,
            "label":             label,
            "prompt_wo_context": prompt_wo_context,
            "prompt_w_context":  prompt_w_context,
        })

    # ── Statistics ─────────────────────────────────────────────────────────
    n_case1 = sum(1 for r in results if r["case"] == 1)
    n_case3 = sum(1 for r in results if r["case"] == 3)
    print(f"\nCase filtering complete:")
    print(f"  Total processed : {len(dataset)}")
    print(f"  Case 1 (label 0, non-hallucination) : {n_case1}")
    print(f"  Case 3 (label 1, hallucination)     : {n_case3}")
    print(f"  Total kept      : {len(results)}")

    return results


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_cases(cases: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} cases → {path}")


def load_cases(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: Optional[str] = None):
    """Load a base (non-instruct) causal LM and its tokenizer."""
    name = model_name or config.MODEL_NAME
    print(f"Loading model: {name}")

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" handles multi-GPU / CPU offload automatically.
    # Do NOT call .to() afterwards — it conflicts with device_map.
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer