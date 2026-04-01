import os
import numpy as np
import torch
from tqdm import tqdm
import config

from Typing import Dict, List, Tuple, Optional, Any

def find_answer_token_position(tokenizer, prompt: str, answer: str) -> int:
    full_text = prompt + " " + answer
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    # prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    # full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    
    start_pos = len(prompt_ids)
    end_pos = len(full_ids) # - 1 # Exclude?? the last token which is likely to be a special token (e.g., EOS)
 
    start_pos = min(start_pos, len(full_ids))  # Ensure start_pos is within bounds
    
    return len(prompt_ids)

def pool_hidden_states(hidden_states_at_layer: torch.Tensor, start_pos: int, end_pos: int, method: str = "first") -> torch.Tensor:
    # Use the first token's hidden state as the answer representation for default setting, but we can also experiment with mean pooling, last token, or even all tokens (e.g., concatenation or attention-based pooling) in later phases.
    # Options: "first" | "mean" | "last" | "all"
    
    ans_span = hidden_states_at_layer[start_pos:end_pos]  # Shape: (span_length, hidden_size)
    
    if method == "first":
        return ans_span[0]  # Shape: (hidden_size,)
    elif method == "mean":
        return ans_span.mean(dim=0)  # Shape: (hidden_size,)
    elif method == "last":
        return ans_span[-1]  # Shape: (hidden_size,)
    elif method == "all":
        return ans_span.flatten()  # Shape: (span_length * hidden_size,)
    else:
        raise ValueError(f"Invalid pooling method: {method}")
    
def extract_hidden_states_single(model, tokenizer, prompt: str, answer: str, strategy: str = "first") -> torch.Tensor:
    full_text = prompt + " " + answer
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # hidden_states is a tuple of (num_layers, batch_size, seq_length, hidden_size)
    all_hidden = outputs.hidden_states  # Tuple of hidden states at each layer

    # Extract the relevant hidden states for the answer span
    answer_start = find_answer_token_position(tokenizer, prompt, answer)
    answer_end = answer_start + len(tokenizer(answer, add_special_tokens=False).input_ids)

    layer_vectors = []
    for layer_hs in all_hidden:
        hs_2d = layer_hs[0]  # Shape: (seq_length, hidden_size)
        layer_vector = pool_hidden_states(hs_2d, answer_start, answer_end, method=strategy)
        layer_vectors.append(layer_vector)
        
    return torch.stack(layer_vectors)  # Shape: (num_layers, hidden_size) or (num_layers, span_length * hidden_size) if method="all"

def extract_all_hs(model, tokenizer, cases: List[Dict[str, Any]], strategy: str = "first") -> List[torch.Tensor]:
    all_hs = []
    all_labels = []
    
    model.eval()
    
    for item in tqdm(cases, desc="Extracting hidden states [(Strategy: {strategy})]"):
        label = item["label"]
        prompt = item["prompt_w_context"] if label == 0 else item["prompt_wo_context"]  # Use the prompt corresponding to the label (context-dependent vs context-independent)
        answer = item["ans_w_context"] if label == 0 else item["ans_wo_context"]
        
        if not answer.strip():
            continue  # Skip cases where the answer is empty
        
        try:
            hs = extract_hidden_states_single(model, tokenizer, prompt, answer, strategy=strategy)
            all_hs.append(hs)
            all_labels.append(label)
            
        except Exception as e:
            print(f"Error processing case with question: {item['question']}. Error: {e}")
            continue
        
    return torch.stack(all_hs), torch.tensor(all_labels)  # Shapes: (num_cases, num_layers, hidden_size), (num_cases,)

def save_hidden_states(hidden_states: torch.Tensor, labels: torch.Tensor, strategy: str):
    os.makedirs(config.HIDDEN_STATE_DIR, exist_ok=True)
    hs_path = os.path.join(config.HIDDEN_STATE_DIR, f"hidden_states_{strategy}.pt, .npy") # 둘 다 저장해두고싶음
    labels_path = os.path.join(config.HIDDEN_STATE_DIR, f"labels_{strategy}.pt, .npy") # ㅇㅒ도 수정

    torch.save(hidden_states.cpu(), hs_path)
    torch.save(labels.cpu(), labels_path)
    
    print(f"Hidden states saved to {hs_path} | Shape: {hidden_states.shape}")
    print(f"Labels saved to {labels_path} | Shape: {labels.shape}")

def load_hidden_states(output_dir: str, strategy: str) -> Tuple[torch.Tensor, torch.Tensor]:
    hs_path = os.path.join(config.HIDDEN_STATE_DIR, f"hidden_states_{strategy}.pt")
    labels_path = os.path.join(config.HIDDEN_STATE_DIR, f"labels_{strategy}.pt")

    hidden_states = torch.load(hs_path)
    labels = torch.load(labels_path)

    # Below, would it be better to use try except or if else
    if hidden_states.shape == save_hidden_states(hidden_states).shape:
        print(f"Hidden states loaded from {hs_path} | Shape: {hidden_states.shape}")
        print(f"Labels loaded from {labels_path} | Shape: {labels.shape}")
    else:
        print(f"Warning: Loaded hidden states shape {hidden_states.shape} does not match expected shape {save_hidden_states(hidden_states).shape}")
        print(f"Warning: Loaded labels shape {labels.shape} does not match expected shape {save_hidden_states(labels).shape}")
        
    return hidden_states, labels