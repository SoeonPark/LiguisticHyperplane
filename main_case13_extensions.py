"""
main_case13_extensions.py

Additional experiments for Case 1 vs Case 3:
1. label-balanced sampling probe
2. across-layer sequence classifier (BiLSTM / small Transformer)
"""

import argparse
import os
import sys

_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg in ("--gpu", "-g") and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import numpy as np

import case13_result_summary as c13s
import config
import data_utils as du
import extract_hidden_state as ehs
import layer_sequence_model as lsm
import sampled_layer_probe as slp
from extended_experiment_utils import (
    build_paths,
    cleanup_memory,
    strategies_to_run,
)


def _expected_hidden_state_metadata(args, strategy: str, num_cases: int) -> dict:
    return {
        "model_name": args.model,
        "prompt_source": "prompt_w_context",
        "answer_source": "ans_w_context",
        "num_cases": num_cases,
        "max_new_tokens": config.MAX_NEW_TOKENS,
    }


def phase_data(paths: dict, args) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 0: Data filtering  [{paths['exp_name']}]")
    print(
        f"  subset={args.subset}  split={args.data_split}  "
        f"max_samples={args.max_samples or 'all'}"
    )
    print("=" * 60)

    if os.path.exists(paths["cases"]) and not args.force_recompute:
        print(f"[skip] Already exists: {paths['cases']}")
        return

    model, tokenizer = du.load_model_and_tokenizer(args.model)
    try:
        cases_all = du.run_case_filtering(
            model,
            tokenizer,
            max_samples=args.max_samples,
            answer_types=args.answer_types,
            subset=args.subset,
            data_split=args.data_split,
        )
        du.save_cases(cases_all, paths["cases_all"])
        filtered = [case for case in cases_all if case["case"] in [1, 3]]
        du.save_cases(filtered, paths["cases"])
    finally:
        del model
        del tokenizer
        cleanup_memory()


def phase_extract(paths: dict, args, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 1: Extract hidden states  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)

    hs_path = os.path.join(paths["hs_dir"], f"hs_{strategy}.npy")
    cases = du.load_cases(paths["cases"])
    expected_meta = _expected_hidden_state_metadata(args, strategy, num_cases=len(cases))

    if not args.force_recompute and os.path.exists(hs_path):
        is_current, reason = ehs.hidden_state_cache_is_current(
            strategy,
            hs_dir=paths["hs_dir"],
            expected_metadata=expected_meta,
        )
        if is_current:
            print(f"[skip] Reusing valid hidden states: {hs_path}")
            return
        print(f"[recompute] Hidden-state cache invalid for strategy={strategy}: {reason}")

    model, tokenizer = du.load_model_and_tokenizer(args.model)
    try:
        hidden_states, labels = ehs.extract_all_hidden_states(
            model,
            tokenizer,
            cases,
            strategy=strategy,
        )
        ehs.save_hidden_states(
            hidden_states,
            labels,
            strategy=strategy,
            out_dir=paths["hs_dir"],
            metadata=expected_meta,
        )
    finally:
        del model
        del tokenizer
        cleanup_memory()


def phase_probe_sampled(paths: dict, args, strategy: str, sampling_scope: str) -> None:
    print("\n" + "=" * 60)
    print(
        f"Phase 2: Sampled layer probe  [{paths['exp_name']}]  "
        f"strategy={strategy}  scope={sampling_scope}  sampling={args.sampling_method}"
    )
    print("=" * 60)

    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    num_samples = int(hidden_states.shape[0])
    if sampling_scope == "dataset" and args.sampling_method != "none":
        counts = np.unique(labels, return_counts=True)[1]
        num_samples = int(min(counts) * 2)
    expected_metadata = {
        "sampling_method": args.sampling_method,
        "sampling_scope": sampling_scope,
        "probe_test_size": config.PROBE_TEST_SIZE,
        "probe_max_iter": config.PROBE_MAX_ITER,
        "random_seed": config.RANDOM_SEED,
        "num_samples": num_samples,
        "num_layers": int(hidden_states.shape[1]),
        "hidden_dim": int(hidden_states.shape[2]),
    }

    if not args.force_recompute:
        is_current, reason = slp.sampled_probe_cache_is_current(
            strategy,
            args.sampling_method,
            sampling_scope,
            probe_dir=paths["probe_sampled_dir"],
            expected_metadata=expected_metadata,
        )
        if is_current:
            print("[skip] Reusing sampled probe results.")
            payload = slp.load_sampled_probe_results(
                strategy,
                args.sampling_method,
                sampling_scope,
                probe_dir=paths["probe_sampled_dir"],
            )
            slp.print_summary(payload)
            return
        print(f"[recompute] Sampled probe cache invalid: {reason}")

    cases = du.load_cases(paths["cases"])
    payload = slp.train_sampled_probe_per_layer(
        hidden_states,
        labels,
        cases=cases,
        sampling_method=args.sampling_method,
        sampling_scope=sampling_scope,
    )
    slp.save_sampled_probe_results(
        payload,
        strategy,
        args.sampling_method,
        sampling_scope,
        probe_dir=paths["probe_sampled_dir"],
    )
    slp.print_summary(payload)


def phase_sequence_model(
    paths: dict,
    args,
    strategy: str,
    model_type: str,
    sampling_scope: str,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"Phase 3: Sequence classifier  [{paths['exp_name']}]  "
        f"strategy={strategy}  model={model_type}  "
        f"scope={sampling_scope}  sampling={args.sampling_method}"
    )
    print("=" * 60)

    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    num_samples = int(hidden_states.shape[0])
    if sampling_scope == "dataset" and args.sampling_method != "none":
        counts = np.unique(labels, return_counts=True)[1]
        num_samples = int(min(counts) * 2)
    model_config = {
        "sampling_method": args.sampling_method,
        "sampling_scope": sampling_scope,
        "test_size": config.PROBE_TEST_SIZE,
        "val_size": args.sequence_val_size,
        "random_seed": config.RANDOM_SEED,
        "epochs": args.sequence_epochs,
        "batch_size": args.sequence_batch_size,
        "learning_rate": args.sequence_lr,
        "weight_decay": args.sequence_weight_decay,
        "patience": args.sequence_patience,
        "dropout": args.sequence_dropout,
        "model_dim": args.sequence_model_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_layers": args.lstm_layers,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_ff_dim": args.transformer_ff_dim,
    }

    expected_metadata = {
        "model_type": model_type,
        "sampling_method": args.sampling_method,
        "sampling_scope": sampling_scope,
        "input_shape": {
            "num_samples": num_samples,
            "num_layers": int(hidden_states.shape[1]),
            "hidden_dim": int(hidden_states.shape[2]),
        },
        "config": model_config,
    }

    if not args.force_recompute:
        is_current, reason = lsm.sequence_result_cache_is_current(
            strategy,
            model_type,
            args.sampling_method,
            sampling_scope,
            save_dir=paths["sequence_dir"],
            expected_metadata=expected_metadata,
        )
        if is_current:
            print("[skip] Reusing sequence-model results.")
            payload = lsm.load_sequence_results(
                strategy,
                model_type,
                args.sampling_method,
                sampling_scope,
                save_dir=paths["sequence_dir"],
            )
            lsm.print_sequence_summary(payload)
            return
        print(f"[recompute] Sequence-model cache invalid: {reason}")

    cases = du.load_cases(paths["cases"])
    payload = lsm.train_layer_sequence_model(
        hidden_states,
        labels,
        cases=cases,
        model_type=model_type,
        model_config=model_config,
    )
    lsm.save_sequence_results(
        payload,
        strategy,
        model_type,
        args.sampling_method,
        sampling_scope,
        save_dir=paths["sequence_dir"],
    )
    lsm.print_sequence_summary(payload)


def phase_summary(paths: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 4: Result summary  [{paths['exp_name']}]")
    print("=" * 60)
    summary_paths = c13s.build_case13_summary(
        probe_dir=paths["probe_sampled_dir"],
        sequence_dir=paths["sequence_dir"],
        summary_dir=paths["summary_dir"],
    )
    print(f"  CSV  : {summary_paths['csv_path']}")
    print(f"  Plot : {summary_paths['plot_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Additional Case 1 vs Case 3 experiments"
    )
    parser.add_argument("--model", type=str, default=config.MODEL_NAME)
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for output folder")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "data", "extract", "probe_sampled", "sequence_model", "summary"],
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="first",
        choices=["first", "mean", "last", "all"],
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="both",
        choices=["fullwiki", "distractor", "both"],
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="validation",
        choices=["train", "validation"],
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--answer_types", type=str, nargs="+", default=config.ANSWER_TYPES)
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="undersample",
        choices=["none", "undersample"],
        help="How to balance Case 1/Case 3 labels on the training split.",
    )
    parser.add_argument(
        "--sampling_scope",
        type=str,
        default="train",
        choices=["train", "dataset", "both"],
        help="Balance only the train split or the full dataset before splitting.",
    )
    parser.add_argument(
        "--sequence_model",
        type=str,
        default="both",
        choices=["bilstm", "transformer", "both"],
    )
    parser.add_argument("--sequence_epochs", type=int, default=20)
    parser.add_argument("--sequence_batch_size", type=int, default=32)
    parser.add_argument("--sequence_lr", type=float, default=1e-3)
    parser.add_argument("--sequence_weight_decay", type=float, default=1e-4)
    parser.add_argument("--sequence_patience", type=int, default=5)
    parser.add_argument("--sequence_val_size", type=float, default=0.1)
    parser.add_argument("--sequence_dropout", type=float, default=0.1)
    parser.add_argument("--sequence_model_dim", type=int, default=256)
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_ff_dim", type=int, default=512)
    parser.add_argument("--gpu", "-g", type=str, default="0")
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Ignore cached hidden states / additional experiment results and recompute.",
    )

    args = parser.parse_args()
    paths = build_paths(
        args.model,
        args.subset,
        args.data_split,
        args.max_samples,
        args.answer_types,
        args.tag,
    )
    for key, val in paths.items():
        if key not in ("cases", "cases_all", "exp_name") and not val.endswith(".json"):
            os.makedirs(val, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Model          : {args.model}")
    print(f"  Subset         : {args.subset}")
    print(f"  Split          : {args.data_split}  (max_samples={args.max_samples or 'all'})")
    print(f"  Strategy       : {args.strategy}")
    print(f"  Sampling       : {args.sampling_method}")
    print(f"  Sampling scope : {args.sampling_scope}")
    print(f"  Sequence model : {args.sequence_model}")
    print(f"  Force          : {args.force_recompute}")
    print(f"  Experiment     : {paths['exp_name']}")
    print(f"  GPU            : {_gpu}")
    print(f"{'=' * 60}")

    strategies = strategies_to_run(args.strategy)
    sequence_models = ["bilstm", "transformer"] if args.sequence_model == "both" else [args.sequence_model]
    sampling_scopes = ["train", "dataset"] if args.sampling_scope == "both" else [args.sampling_scope]

    if args.phase == "all":
        phase_data(paths, args)
        for strategy in strategies:
            phase_extract(paths, args, strategy)
            for sampling_scope in sampling_scopes:
                phase_probe_sampled(paths, args, strategy, sampling_scope)
                for model_type in sequence_models:
                    phase_sequence_model(paths, args, strategy, model_type, sampling_scope)
        phase_summary(paths)

    elif args.phase == "data":
        phase_data(paths, args)

    elif args.phase == "extract":
        for strategy in strategies:
            phase_extract(paths, args, strategy)

    elif args.phase == "probe_sampled":
        for strategy in strategies:
            for sampling_scope in sampling_scopes:
                phase_probe_sampled(paths, args, strategy, sampling_scope)

    elif args.phase == "sequence_model":
        for strategy in strategies:
            for sampling_scope in sampling_scopes:
                for model_type in sequence_models:
                    phase_sequence_model(paths, args, strategy, model_type, sampling_scope)

    elif args.phase == "summary":
        phase_summary(paths)


if __name__ == "__main__":
    main()
