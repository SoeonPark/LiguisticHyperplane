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
    para_dict = {title: "".join(sents) for title, sents in paragraphs}
    context_parts = []
    seen = set()
    for title in supporting_facts_titles:
        if title not in seen and title in para_dict:
            context_parts.append(para_dict[title])
            seen.add(title)
    return " ".join(context_parts)


# Extract Answer and Supporting Facts from HotpotQA
def extract_answer(generated_answer: str, prompt: str) -> str:
    # Remove the prompt from the generated answer
    if generated_answer.startswith(prompt):
        return generated_answer[len(prompt):].strip()
    return generated_answer.strip()

def normalize_answer(answer: str) -> str:
    return answer.lower().strip()

def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold) or normalize(gold) in normalize(pred)

# Hallucination Types Classifier
def classify_case(correct_w_context: bool, correct_wo_context: bool, 
                           ans_w_context: str, ans_wo_context: str) -> int:
    """
    Returns:
        - Case 1: Answer correct with context, incorrect without context (hallucination type 1) -- Proof of LLM relying on context to answer correctly, which is a form of hallucination when the context is fabricated or irrelevant.
        - Case 2: Answer incorrect with context, correct without context (hallucination type 2)
        - Case 3: Answer incorrect in both cases with the same answer (hallucination type 3) -- Indicates that the model is consistently hallucinating the same information regardless of context.
        - Case 4: Answer incorrect in both cases with different answers (hallucination type 4)
    """
    if correct_w_context and not correct_wo_context:
        return 1
    elif not correct_w_context and correct_wo_context:
        return 2
    elif not correct_w_context and not correct_wo_context:
        if normalize_answer(ans_w_context) == normalize_answer(ans_wo_context):
            return 3
        else:
            return 4
    return 0  
    
def run_case_filtering(model:str, tokenizer:str, max_samples: Optional[int] = None) -> Dict[str, List[Dict[str, str]]]:
    print("Loading HotpotQA...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=config.DATA_SPLIT) 
    
    if max_samples is not None:
        dataset = dataset.select(range((min(max_samples, len(dataset)))))
        
    results = []
    model.eval()
    
    for item in tqdm(dataset):
        question = item["question"]
        answer = item["answer"]
        supporting_facts = item["supporting_facts"]
        qtype = item["type"]
        level = item["level"]
        
        if qtype != "comparison": # bridge
            
        elif qtype == "comparison": # comparison
            if gold.lower() not in ["yes", "no"]: # For short answer type
                answer = "entity"
            else:
                answer = gold.lower()
        elif answer_type not in config.ANSWER_TYPES:
            continue
        
        # Context Construction
        supporting_facts_titles = [sf[0] for sf in supporting_facts]
        paragraphs = list(zip(item["context"]["title"], item["context"]["sentences"]))
        context = build_context_string(supporting_facts_titles, paragraphs)
        
        prompt_wo_context = build_prompt_no_context(question)
        prompt_w_context = build_prompt_with_context(question, context)
        
        # Inference
        ans_wo_context = generate_answer(model, tokenizer, prompt_wo_context)
        ans_w_context = generate_answer(model, tokenizer, prompt_w_context)
        
        # Classification
        correct_wo_context = is_correct(ans_wo_context, answer)
        correct_w_context = is_correct(ans_w_context, answer)
        case = classify_case(correct_w_context, correct_wo_context, ans_w_context, ans_wo_context)
        
        if case not in [1, 3]:  # We focus on cases where hallucination is present (case 1 and case 3)
            # 근데 살펴봐야될 것 같긴한데 음 따로 저장해둬야되나
            continue        
        
        label = 0 if case == 1 else 1  # 0 : Non-hallucination (context-dependent), 1 : Hallucination (context-independent)
        
        results.append({
            "question": question,
            "answer": answer,
            "gold_answer": gold,
            "supporting_facts": supporting_facts,
            "context": context,
            "ans_wo_context": ans_wo_context,
            "ans_w_context": ans_w_context,
            "case": case,
            "label": label,
            "qtype": qtype,
            "prompt_wo_context": prompt_wo_context,
            "prompt_w_context": prompt_w_context,
        })

    print(f"Statistics:")
    print(f"Total cases processed: {len(dataset)}")
    print(f"Case 1 (Context-dependent correct): {sum(1 for r in results if r['case'] == 1)}")
    print(f"Case 3 (Context-independent hallucination): {sum(1 for r in results if r['case'] == 3)}")
    print(f"Total Cases: {len(results)}")

    return results
   
def generate_answer(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,  # Ensure padding is handled correctly
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_answer(generated_text, prompt)

# Save and Load Cases
def save_cases(cases: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} cases → {path}")

def load_cases(path: str) -> list:
    with open(path) as f:
        return json.load(f)

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