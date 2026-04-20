"""
Integrated runner for answer-token alignment and probe classification.

Phases:
  align    : verify exact answer-start token slicing.
  extract  : extract hidden states at the aligned answer span.
  probe    : train/evaluate a linear probe over cached hidden states.
  classify : apply a saved probe to cached hidden states.
  all      : align + extract + probe + classify.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import config
import extract_hidden_state as ehs


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Saved JSON -> {path}")


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved JSONL -> {path}")


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_tokenizer(model_name: str, tokenizer_name: Optional[str] = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.model_name, args.tokenizer_name)
    kwargs = {}
    if args.device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        if args.num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            n_gpus = int(args.num_gpus)
            if n_gpus != 1:
                kwargs["device_map"] = "auto"
                kwargs["max_memory"] = {i: f"{args.max_gpu_memory}GiB" for i in range(n_gpus)}
    elif args.device != "cpu":
        raise ValueError("device must be cuda or cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, **kwargs)
    if args.device == "cuda" and args.num_gpus == "1":
        model.cuda()
    elif args.device == "cpu":
        model.cpu()
    model.eval()
    return model, tokenizer


def load_label_map(path: Optional[str], label_key: str) -> Dict[int, int]:
    if not path:
        return {}
    rows = read_jsonl(path) if path.endswith(".jsonl") else read_json(path)
    label_map = {}
    for idx, row in enumerate(rows):
        input_index = int(row.get("input_index", idx))
        if label_key in row and row[label_key] is not None:
            label_map[input_index] = int(row[label_key])
    return label_map


def load_cases(args) -> List[Dict]:
    if args.cases_json:
        cases = read_json(args.cases_json)
        return cases[:args.max_samples] if args.max_samples is not None else cases

    if not args.data_path:
        raise ValueError("provide --cases-json or --data-path")

    rows = read_jsonl(args.data_path)
    pred_rows = read_jsonl(args.pred_path) if args.pred_path else []
    pred_by_index = {int(row["input_index"]): row for row in pred_rows}
    label_map = load_label_map(args.labels_path, args.label_key)

    cases = []
    seen = set()
    for row in rows:
        input_index = int(row["input_index"])
        if input_index in seen:
            continue
        if int(row.get("assigned_process", args.eval_assigned_process)) != args.eval_assigned_process:
            continue
        seen.add(input_index)

        pred = pred_by_index.get(input_index)
        answer = pred.get("string", [""])[0] if pred is not None else row.get("gold_answers", "")
        case = {
            "input_index": input_index,
            "prompt_w_context": row.get(args.prompt_key, row.get("context_string", "")),
            "ans_w_context": answer,
            "gold_answer": row.get("gold_answers", ""),
            "article": row.get("article", ""),
        }
        if input_index in label_map:
            case["label"] = int(label_map[input_index])
        elif args.label_key in row and row[args.label_key] is not None:
            case["label"] = int(row[args.label_key])
        cases.append(case)

    return cases[:args.max_samples] if args.max_samples is not None else cases


def case_prompt_answer(case: Dict, prompt_key: str, answer_key: str) -> Tuple[str, str]:
    if prompt_key in case and answer_key in case:
        return case[prompt_key], case[answer_key]
    return case["prompt_w_context"], case["ans_w_context"]


def build_alignment_report(args, cases: List[Dict]) -> List[Dict]:
    tokenizer = load_tokenizer(args.model_name, args.tokenizer_name)
    records = []
    failures = 0
    for idx, case in enumerate(tqdm(cases, desc="Checking answer-token alignment")):
        prompt, answer = case_prompt_answer(case, args.prompt_key, args.answer_key)
        try:
            _, alignment = ehs.align_answer_span(tokenizer, prompt, answer)
            record = {
                "status": "ok",
                "case_index": idx,
                "input_index": case.get("input_index"),
                "label": case.get(args.label_key),
                **alignment.to_json_dict(),
            }
            if idx < args.preview:
                first = record["answer_tokens"][0] if record["answer_tokens"] else None
                print(
                    f"[align:{idx}] input_index={record.get('input_index')} "
                    f"start={record['answer_start']} end={record['answer_end']} "
                    f"len={record['answer_len']} first_token={first!r}"
                )
                print(f"  decoded_span={record['decoded_answer_span'][:200]!r}")
        except Exception as exc:
            failures += 1
            record = {
                "status": "error",
                "case_index": idx,
                "input_index": case.get("input_index"),
                "label": case.get(args.label_key),
                "error": str(exc),
                "prompt_preview": prompt[:300],
                "answer_preview": answer[:300],
            }
            print(f"[align-error:{idx}] {exc}")
        records.append(record)
    print(f"Alignment summary: ok={len(records) - failures}, failed={failures}, total={len(records)}")
    return records


def phase_align(args, cases: List[Dict]) -> None:
    records = build_alignment_report(args, cases)
    ehs.write_alignment_report(records, args.alignment_report)


def phase_extract(args, cases: List[Dict]) -> None:
    model, tokenizer = load_model_and_tokenizer(args)
    try:
        hidden_states, labels, alignments = ehs.extract_all_hidden_states(
            model,
            tokenizer,
            cases,
            strategy=args.strategy,
            prompt_key=args.prompt_key,
            answer_key=args.answer_key,
            label_key=args.label_key,
            max_samples=args.max_samples,
            return_alignments=True,
        )
        ehs.save_hidden_states(
            hidden_states,
            labels,
            strategy=args.strategy,
            out_dir=args.hs_dir,
            metadata={
                "model_name": args.model_name,
                "prompt_key": args.prompt_key,
                "answer_key": args.answer_key,
                "label_key": args.label_key,
                "source": args.cases_json or args.data_path,
                "pred_path": args.pred_path,
            },
        )
        ehs.write_alignment_report(alignments, args.alignment_report)
    finally:
        del model
        del tokenizer
        cleanup_memory()


def safe_roc_auc(y_true, proba) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def train_probe_per_layer(args, hidden_states: np.ndarray, labels: np.ndarray):
    if labels is None:
        raise ValueError("labels are required for probe training")
    idx = np.arange(labels.shape[0])
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.probe_test_size,
        random_state=args.random_seed,
        stratify=stratify,
    )

    results = []
    best = None
    for layer_idx in tqdm(range(hidden_states.shape[1]), desc="Training probe per layer"):
        x = hidden_states[:, layer_idx, :]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=args.probe_max_iter,
                random_state=args.random_seed,
                solver="lbfgs",
                class_weight=args.class_weight,
            ),
        )
        model.fit(x[train_idx], labels[train_idx])
        train_pred = model.predict(x[train_idx])
        test_pred = model.predict(x[test_idx])
        test_proba = model.predict_proba(x[test_idx])[:, 1]
        auroc = safe_roc_auc(labels[test_idx], test_proba)
        row = {
            "layer": int(layer_idx),
            "accuracy": round(float(accuracy_score(labels[test_idx], test_pred)), 4),
            "auroc": round(float(auroc), 4) if auroc is not None else None,
            "train_acc": round(float(accuracy_score(labels[train_idx], train_pred)), 4),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        }
        results.append(row)
        score = -1.0 if auroc is None else auroc
        if best is None or score > best["score"]:
            best = {"score": score, "layer": int(layer_idx), "model": model, "metrics": row}
    return results, best


def save_probe(args, results: List[Dict], best: Dict) -> str:
    os.makedirs(args.probe_dir, exist_ok=True)
    metrics_path = os.path.join(args.probe_dir, f"probe_{args.strategy}.json")
    model_path = os.path.join(args.probe_dir, f"probe_{args.strategy}_best.pkl")
    write_json(
        {
            "strategy": args.strategy,
            "best_layer": best["layer"],
            "best_metrics": best["metrics"],
            "results": results,
            "probe_test_size": args.probe_test_size,
            "probe_max_iter": args.probe_max_iter,
            "random_seed": args.random_seed,
            "class_weight": args.class_weight,
        },
        metrics_path,
    )
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "strategy": args.strategy,
                "layer": best["layer"],
                "model": best["model"],
                "metrics": best["metrics"],
                "all_results": results,
            },
            f,
        )
    print(f"Saved best probe -> {model_path}")
    return model_path


def phase_probe(args) -> str:
    hidden_states, labels = ehs.load_hidden_states(args.strategy, hs_dir=args.hs_dir)
    if labels is None:
        raise ValueError("labels unavailable; use labeled cases or --labels-path before probe")
    vals, counts = np.unique(labels, return_counts=True)
    print(f"Probe labels: {dict(zip([int(v) for v in vals], [int(c) for c in counts]))}")
    results, best = train_probe_per_layer(args, hidden_states, labels)
    print(f"Best layer={best['layer']} AUROC={best['metrics']['auroc']} Acc={best['metrics']['accuracy']}")
    return save_probe(args, results, best)


def phase_classify(args) -> None:
    probe_model = args.probe_model or os.path.join(args.probe_dir, f"probe_{args.strategy}_best.pkl")
    with open(probe_model, "rb") as f:
        payload = pickle.load(f)
    hidden_states, labels = ehs.load_hidden_states(args.strategy, hs_dir=args.hs_dir)
    layer = int(payload["layer"])
    model = payload["model"]
    x = hidden_states[:, layer, :]
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]

    rows = []
    for idx, (p, prob) in enumerate(zip(pred, proba)):
        row = {
            "case_index": idx,
            "pred_label": int(p),
            "hallucination_probability": float(prob),
            "probe_layer": layer,
        }
        if labels is not None:
            row["gold_label"] = int(labels[idx])
        rows.append(row)

    metrics = {"num_samples": len(rows), "probe_layer": layer}
    if labels is not None:
        metrics["accuracy"] = float(accuracy_score(labels, pred))
        metrics["auroc"] = safe_roc_auc(labels, proba)
        print(f"Classification test: Acc={metrics['accuracy']:.4f} AUROC={metrics['auroc']}")
    else:
        print("Classification test: labels unavailable; wrote probabilities only")

    write_jsonl(rows, args.classification_output)
    write_json(metrics, args.classification_output + ".metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Run realtime hallucination alignment/probe checks.")
    parser.add_argument("--phase", choices=["align", "extract", "probe", "classify", "all"], default="all")
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=config.DEVICE)
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max-gpu-memory", type=int, default=27)
    parser.add_argument("--cases-json", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--pred-path", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument("--eval-assigned-process", type=int, default=0)
    parser.add_argument("--prompt-key", type=str, default="prompt_w_context")
    parser.add_argument("--answer-key", type=str, default="ans_w_context")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strategy", choices=["prompt_last", "first", "mean", "last"], default="first")
    parser.add_argument("--out-dir", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--hs-dir", type=str, default=config.HIDDEN_STATE_DIR)
    parser.add_argument("--probe-dir", type=str, default=config.PROBE_MODEL_DIR)
    parser.add_argument("--alignment-report", type=str, default=os.path.join(config.OUTPUT_DIR, "alignment_report.jsonl"))
    parser.add_argument("--classification-output", type=str, default=os.path.join(config.OUTPUT_DIR, "classification.jsonl"))
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--probe-test-size", type=float, default=config.PROBE_TEST_SIZE)
    parser.add_argument("--probe-max-iter", type=int, default=config.PROBE_MAX_ITER)
    parser.add_argument("--random-seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    parser.add_argument("--probe-model", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.hs_dir, exist_ok=True)
    os.makedirs(args.probe_dir, exist_ok=True)

    cases = None
    if args.phase in {"align", "extract", "all"}:
        cases = load_cases(args)
        print(f"Loaded cases: {len(cases)}")

    if args.phase == "align":
        phase_align(args, cases)
    elif args.phase == "extract":
        phase_extract(args, cases)
    elif args.phase == "probe":
        phase_probe(args)
    elif args.phase == "classify":
        phase_classify(args)
    elif args.phase == "all":
        phase_align(args, cases)
        phase_extract(args, cases)
        args.probe_model = phase_probe(args)
        phase_classify(args)


if __name__ == "__main__":
    main()
