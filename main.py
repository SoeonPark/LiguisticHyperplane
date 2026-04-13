"""
main.py - minimal fix: phase_token_pos 항상 ALL_STRATEGIES 고정
"""

import argparse
import gc
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
import layer_analysis as la
import layer_sequence_model as lsm
import linear_probe as lp
import sampled_layer_probe as slp
import analysis_extended as ae
import torch

ALL_STRATEGIES = ["first", "mean", "last"]


def make_model_short(model_name: str) -> str:
    short = model_name.split("/")[-1].lower()
    short = short.replace("meta-llama-", "llama").replace("meta-", "")
    short = short.replace("mistralai-", "").replace("microsoft-", "")
    return short


def build_paths(
    model_name: str,
    subset: str,
    data_split: str,
    max_samples: int | None,
    answer_types: list[str],
    tag: str,
) -> dict:
    short    = make_model_short(model_name)
    split_s  = "val" if data_split == "validation" else data_split
    sample_s = f"n{max_samples}" if max_samples is not None else "nall"
    type_s   = "-".join(sorted(answer_types))
    gen_s    = f"gen{config.MAX_NEW_TOKENS}"
    exp_name = f"{short}__{subset}_{split_s}__{sample_s}__types_{type_s}__{gen_s}"
    if tag:
        exp_name += f"__{tag}"
    base = os.path.join("outputs", exp_name)
    return {
        "base": base,
        "cases": os.path.join(base, "cases.json"),
        "cases_all": os.path.join(base, "cases_all.json"),
        "hs_dir": os.path.join(base, "hidden_states"),
        "tokenwise_dir": os.path.join(base, "tokenwise"),
        "probe_dir": os.path.join(base, "probe_results"),
        "probe_sampled_dir": os.path.join(base, "probe_results_sampled"),
        "sequence_dir": os.path.join(base, "sequence_results"),
        "summary_dir": os.path.join(base, "summary"),
        "figure_dir": os.path.join(base, "figures"),
        "log_dir": os.path.join(base, "logs"),
        "exp_name": exp_name,
    }


def _strategies_to_run(strategy_arg: str):
    """--strategy all  →  ['first', 'mean', 'last'],  else single."""
    return ALL_STRATEGIES if strategy_arg == "all" else [strategy_arg]


def _sequence_models_to_run(model_arg: str) -> list[str]:
    return ["bilstm", "transformer"] if model_arg == "both" else [model_arg]


def _sampling_scopes_to_run(scope_arg: str) -> list[str]:
    return ["train", "dataset"] if scope_arg == "both" else [scope_arg]


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _expected_hidden_state_metadata(args, strategy: str, num_cases: int) -> dict:
    return {
        "model_name": args.model,
        "prompt_source": "prompt_w_context",
        "answer_source": "ans_w_context",
        "num_cases": num_cases,
        "max_new_tokens": config.MAX_NEW_TOKENS,
    }


def _expected_probe_metadata(hidden_states, hs_meta: dict) -> dict:
    return {
        "balanced": bool(lp.BALANCED),
        "probe_test_size": config.PROBE_TEST_SIZE,
        "probe_max_iter": config.PROBE_MAX_ITER,
        "random_seed": config.RANDOM_SEED,
        "num_samples": int(hidden_states.shape[0]),
        "num_layers": int(hidden_states.shape[1]),
        "hidden_dim": int(hidden_states.shape[2]),
        "source_hs_cache_version": hs_meta["cache_version"],
        "source_hs_strategy": hs_meta["strategy"],
        "source_model_name": hs_meta.get("model_name"),
        "source_num_cases": hs_meta.get("num_cases"),
    }


def _expected_case13_num_samples(
    labels: np.ndarray,
    sampling_scope: str,
    sampling_method: str,
) -> int:
    num_samples = int(labels.shape[0])
    if sampling_scope == "dataset" and sampling_method != "none":
        _, counts = np.unique(labels, return_counts=True)
        if len(counts) >= 2:
            num_samples = int(min(counts) * 2)
    return num_samples


def _expected_sampled_probe_metadata(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    args,
    sampling_scope: str,
) -> dict:
    return {
        "sampling_method": args.sampling_method,
        "sampling_scope": sampling_scope,
        "probe_test_size": config.PROBE_TEST_SIZE,
        "probe_max_iter": config.PROBE_MAX_ITER,
        "random_seed": config.RANDOM_SEED,
        "num_samples": _expected_case13_num_samples(
            labels,
            sampling_scope,
            args.sampling_method,
        ),
        "num_layers": int(hidden_states.shape[1]),
        "hidden_dim": int(hidden_states.shape[2]),
    }


def _build_sequence_model_config(args, sampling_scope: str) -> dict:
    return {
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


def _expected_sequence_metadata(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    args,
    model_type: str,
    sampling_scope: str,
) -> tuple[dict, dict]:
    model_config = _build_sequence_model_config(args, sampling_scope)
    expected_metadata = {
        "model_type": model_type,
        "sampling_method": args.sampling_method,
        "sampling_scope": sampling_scope,
        "input_shape": {
            "num_samples": _expected_case13_num_samples(
                labels,
                sampling_scope,
                args.sampling_method,
            ),
            "num_layers": int(hidden_states.shape[1]),
            "hidden_dim": int(hidden_states.shape[2]),
        },
        "config": model_config,
    }
    return expected_metadata, model_config


# ── Phase 0 ───────────────────────────────────────────────────────────────────
def phase_data(paths: dict, args) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 0: Data filtering  [{paths['exp_name']}]")
    print(f"  subset={args.subset}  split={args.data_split}  max_samples={args.max_samples or 'all'}")
    print("=" * 60)

    if os.path.exists(paths["cases"]) and not args.force_recompute:
        print(f"[skip] Already exists: {paths['cases']}")
        return
    if os.path.exists(paths["cases"]) and args.force_recompute:
        print(f"[recompute] Overwriting cached cases: {paths['cases']}")

    model, tokenizer = du.load_model_and_tokenizer(args.model)
    try:
        cases_all = du.run_case_filtering(
            model, tokenizer,
            max_samples=args.max_samples,
            answer_types=args.answer_types,
            subset=args.subset,
            data_split=args.data_split,
        )
        du.save_cases(cases_all, paths["cases_all"])
        filtered = [c for c in cases_all if c["case"] in [1, 3]]
        du.save_cases(filtered, paths["cases"])
    finally:
        del model
        del tokenizer
        _cleanup_memory()

    total = len(cases_all)
    print(f"\nCase distribution (total={total}):")
    for c in range(5):
        n = sum(1 for x in cases_all if x["case"] == c)
        print(f"  Case {c}: {n:>5}  ({n/total*100:.1f}%)")
    print(f"  Filtered cases (1 & 3): {len(filtered)}  ({len(filtered)/total*100:.1f}%)")


# ── Phase 1 ───────────────────────────────────────────────────────────────────
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
            model, tokenizer, cases, strategy=strategy
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
        _cleanup_memory()
    print("Phase 1 completed. You can now run Phase 2 to train the probes.")


def phase_extract_tokenwise(paths: dict, args) -> None:
    print("\n" + "=" * 60)
    print(f"Phase tokenwise: Extract token-level hidden states  [{paths['exp_name']}]")
    print("=" * 60)

    name = f"tokenwise_w_context_n{args.tokenwise_max_samples or 'all'}"
    payload_path = os.path.join(paths["tokenwise_dir"], f"{name}.pt")
    if os.path.exists(payload_path) and not args.force_recompute:
        print(f"[skip] Already exists: {payload_path}")
        return

    cases = du.load_cases(paths["cases"])
    model, tokenizer = du.load_model_and_tokenizer(args.model)
    try:
        records = ehs.extract_tokenwise_hidden_states(
            model,
            tokenizer,
            cases,
            max_samples=args.tokenwise_max_samples,
        )
        ehs.save_tokenwise_hidden_states(
            records,
            name=name,
            out_dir=paths["tokenwise_dir"],
            metadata={
                "model_name": args.model,
                "prompt_source": "prompt_w_context",
                "answer_source": "ans_w_context",
                "requested_max_samples": args.tokenwise_max_samples,
            },
        )
    finally:
        del model
        del tokenizer
        _cleanup_memory()
    print("Tokenwise extraction completed. This cache can be used for future tuned-lens style analysis.")


# ── Phase 2 ───────────────────────────────────────────────────────────────────
def phase_probe(paths: dict, args, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 2: Linear probe  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)

    result_path = os.path.join(paths["probe_dir"], f"probe_{strategy}.json")
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    hs_meta = ehs.load_hidden_state_metadata(strategy, hs_dir=paths["hs_dir"])
    expected_meta = _expected_probe_metadata(hidden_states, hs_meta)

    if not args.force_recompute and os.path.exists(result_path):
        is_current, reason = lp.probe_cache_is_current(
            strategy,
            probe_dir=paths["probe_dir"],
            expected_metadata=expected_meta,
        )
        if is_current:
            print(f"[skip] Reusing valid probe results: {result_path}")
            results = lp.load_probe_results(strategy, probe_dir=paths["probe_dir"])
            lp.print_summary(results)
            return
        print(f"[recompute] Probe cache invalid for strategy={strategy}: {reason}")

    cases = du.load_cases(paths["cases"])
    results = lp.train_probe_per_layer(hidden_states, labels, cases=cases)
    lp.save_probe_results(
        results,
        strategy,
        probe_dir=paths["probe_dir"],
        metadata=expected_meta,
    )
    lp.print_summary(results)
    print("Phase 2 completed. You can now run Phase 3 to visualize the results.")


# ── Phase 3 ───────────────────────────────────────────────────────────────────
def phase_visualize(paths: dict, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 3: Visualization  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)

    results = lp.load_probe_results(strategy, probe_dir=paths["probe_dir"])
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])

    la.plot_layer_accuracy({strategy: results}, figure_dir=paths["figure_dir"])
    la.plot_tsne(hidden_states, labels, strategy=strategy, figure_dir=paths["figure_dir"])
    print("Phase 3 completed. You can now run Phase 4 to compare token position strategies.")


# ── Phase 4 ───────────────────────────────────────────────────────────────────
def phase_token_pos(paths: dict, args) -> None:
    """
    Always runs all strategies (first, mean, last) for token position comparison,
    since this analysis is specifically about how the choice of token position affects the results.
    """
    print("\n" + "=" * 60)
    print(f"Phase 4: Token position comparison  [{paths['exp_name']}]")
    print("=" * 60)

    results_dict = {}
    for strategy in ALL_STRATEGIES:
        phase_extract(paths, args, strategy)
        phase_probe(paths, args, strategy)
        results_dict[strategy] = lp.load_probe_results(strategy, probe_dir=paths["probe_dir"])
        lp.print_summary(results_dict[strategy])
        _cleanup_memory()

    la.plot_token_position_comparison(results_dict, figure_dir=paths["figure_dir"])
    print("Phase 4 completed. You can now run Phase 5 to analyze probe directions.")


# ── Phase 5: Probe Direction ──────────────────────────────────────────────────
def phase_probe_direction(paths: dict, args, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 5: Probe direction analysis  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)
 
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    model, tokenizer = du.load_model_and_tokenizer(args.model)
    try:
        ae.analyze_probe_direction(
            hidden_states=hidden_states,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            probe_dir=paths["probe_dir"],
            strategy=strategy,
            figure_dir=paths["figure_dir"],
        )
    finally:
        del model
        del tokenizer
        _cleanup_memory()
    print("Phase 5 completed. You can now run Phase 6 for PCA analysis.")
 
 
# ── Phase 6: PCA Analysis ─────────────────────────────────────────────────────
def phase_pca(paths: dict, args, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 6: PCA analysis  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)
 
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
 
    ae.analyze_pca(
        hidden_states=hidden_states,
        labels=labels,
        strategy=strategy,
        figure_dir=paths["figure_dir"],
    )
    print("Phase 6 completed. You can now run Phase 7 for CKA analysis.")
 
 
# ── Phase 7: CKA ─────────────────────────────────────────────────────────────
def phase_cka(paths: dict, args, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 7: CKA analysis  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)
 
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
 
    ae.analyze_cka(
        hidden_states=hidden_states,
        labels=labels,
        strategy=strategy,
        figure_dir=paths["figure_dir"],
    )
    print("Phase 7 completed. You can now run Phase 8 for attention analysis.")
 
 
# ── Phase 8: Attention to Context ─────────────────────────────────────────────
def phase_attention(paths: dict, args) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 8: Attention to context  [{paths['exp_name']}]")
    print("=" * 60)
 
    cases = du.load_cases(paths["cases"])
    model, tokenizer = du.load_model_and_tokenizer(args.model)
    try:
        ae.analyze_attention_to_context(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            figure_dir=paths["figure_dir"],
            max_samples=args.attn_max_samples,
        )
    finally:
        del model
        del tokenizer
        _cleanup_memory()
    print("Phase 8 completed. You can now run Phase 9 to analyze all results together.")
 
# ── Phase analyze_all: 5~8 ALL ──────────────────────────────────────────────
def phase_analyze_all(paths: dict, args, strategies: list) -> None:
    print("\n" + "=" * 60)
    print(f"Phase analyze_all: Running phases 5-8  [{paths['exp_name']}]")
    print("=" * 60)
 
    for strategy in strategies:
        phase_probe_direction(paths, args, strategy)
        phase_pca(paths, args, strategy)
        phase_cka(paths, args, strategy)
 
    phase_attention(paths, args)
    print("Phase analyze_all completed. All analyses are done for the specified strategies.")


# ── Case13: Sampled Probe ────────────────────────────────────────────────────
def phase_probe_sampled(paths: dict, args, strategy: str, sampling_scope: str) -> None:
    print("\n" + "=" * 60)
    print(
        f"Phase case13_probe: Sampled layer probe  [{paths['exp_name']}]  "
        f"strategy={strategy}  scope={sampling_scope}  sampling={args.sampling_method}"
    )
    print("=" * 60)

    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    expected_metadata = _expected_sampled_probe_metadata(
        hidden_states,
        labels,
        args,
        sampling_scope,
    )

    if not args.force_recompute:
        is_current, reason = slp.sampled_probe_cache_is_current(
            strategy,
            args.sampling_method,
            sampling_scope,
            probe_dir=paths["probe_sampled_dir"],
            expected_metadata=expected_metadata,
        )
        breakpoint()
        if is_current:
            print("[skip] Reusing sampled probe results.")
            payload = slp.load_sampled_probe_results(
                strategy,
                args.sampling_method,
                sampling_scope,
                probe_dir=paths["probe_sampled_dir"],
            )
            breakpoint()
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


# ── Case13: Sequence Model ───────────────────────────────────────────────────
def phase_sequence_model(
    paths: dict,
    args,
    strategy: str,
    model_type: str,
    sampling_scope: str,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"Phase case13_sequence: Layer sequence classifier  [{paths['exp_name']}]  "
        f"strategy={strategy}  model={model_type}  "
        f"scope={sampling_scope}  sampling={args.sampling_method}"
    )
    print("=" * 60)

    breakpoint()
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    expected_metadata, model_config = _expected_sequence_metadata(
        hidden_states,
        labels,
        args,
        model_type,
        sampling_scope,
    )

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


# ── Case13: Summary ──────────────────────────────────────────────────────────
def phase_summary(paths: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Phase case13_summary: Result summary  [{paths['exp_name']}]")
    print("=" * 60)

    summary_paths = c13s.build_case13_summary(
        probe_dir=paths["probe_sampled_dir"],
        sequence_dir=paths["sequence_dir"],
        summary_dir=paths["summary_dir"],
    )
    print(f"  CSV  : {summary_paths['csv_path']}")
    print(f"  Plot : {summary_paths['plot_path']}")


def phase_case13_all(paths: dict, args, strategies: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"Phase case13_all: Running sampled probe + sequence models  [{paths['exp_name']}]")
    print("=" * 60)

    phase_data(paths, args)
    sampling_scopes = _sampling_scopes_to_run(args.sampling_scope)
    sequence_models = _sequence_models_to_run(args.sequence_model)

    for strategy in strategies:
        phase_extract(paths, args, strategy)
        for sampling_scope in sampling_scopes:
            phase_probe_sampled(paths, args, strategy, sampling_scope)
            for model_type in sequence_models:
                phase_sequence_model(paths, args, strategy, model_type, sampling_scope)

    phase_summary(paths)
    print("Phase case13_all completed.")

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linguistic Hyperplane Verification Pipeline"
    )
    parser.add_argument("--model", type=str, default=config.MODEL_NAME)
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output folder")
    parser.add_argument("--phase", type=str, default="all",
                        choices=[
                            "all", "data", "extract", "probe", "visualize", "token_pos",
                            "extract_tokenwise", "probe_direction", "pca", "cka",
                            "attention", "analyze_all",
                            "probe_sampled", "sequence_model", "summary", "case13_all"
                        ]
                    )
    parser.add_argument("--strategy", type=str, default="first",
                        choices=["first", "mean", "last", "all"],
                        help="Pooling strategy. 'all' → first+mean+last. token_pos phase는 항상 3개 고정.")
    parser.add_argument("--subset",   type=str, default="both",
                        choices=["fullwiki", "distractor", "both"])
    parser.add_argument("--data_split", type=str, default="validation",
                        choices=["train", "validation"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--answer_types", type=str, nargs="+",
                        default=config.ANSWER_TYPES)
    parser.add_argument("--balanced", action="store_true",
                        help="class_weight='balanced' in LogisticRegression")
    parser.add_argument("--sampling_method", type=str, default="undersample",
                        choices=["none", "undersample"],
                        help="How to balance Case 1/Case 3 labels for sampled probe/sequence-model phases.")
    parser.add_argument("--sampling_scope", type=str, default="train",
                        choices=["train", "dataset", "both"],
                        help="Balance only the train split or the full dataset before splitting.")
    parser.add_argument("--sequence_model", type=str, default="both",
                        choices=["bilstm", "transformer", "both"],
                        help="Which across-layer sequence model to run for case13 phases.")
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
    parser.add_argument("--attn_max_samples", type=int, default=200, help="Max samples for attention analysis (Phase 8). Default 200.")
    parser.add_argument("--tokenwise_max_samples", type=int, default=128,
                        help="Max samples to keep for token-level hidden-state caches used in later lens-style analysis.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore cached cases / hidden states / probe results and recompute.")

    args = parser.parse_args()
    lp.BALANCED = args.balanced

    paths = build_paths(
        args.model,
        args.subset,
        args.data_split,
        args.max_samples,
        args.answer_types,
        args.tag,
    )
    for key, val in paths.items():
        if key not in ("cases", "exp_name") and not val.endswith(".json"):
            os.makedirs(val, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Subset     : {args.subset}")
    print(f"  Split      : {args.data_split}  (max_samples={args.max_samples or 'all'})")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Balanced   : {args.balanced}")
    print(f"  Sampling   : {args.sampling_method}  (scope={args.sampling_scope})")
    print(f"  Seq Model  : {args.sequence_model}")
    print(f"  Force      : {args.force_recompute}")
    print(f"  Experiment : {paths['exp_name']}")
    print(f"  GPU        : {_gpu}")
    print(f"{'='*60}")

    strategies = _strategies_to_run(args.strategy)
    sampling_scopes = _sampling_scopes_to_run(args.sampling_scope)
    sequence_models = _sequence_models_to_run(args.sequence_model)

    if args.phase == "all":
        phase_data(paths, args)
        for s in strategies:
            phase_extract(paths, args, s)
            phase_probe(paths, args, s)
            phase_visualize(paths, s)
        phase_token_pos(paths, args)
        phase_analyze_all(paths, args, strategies)

    elif args.phase == "data":
        phase_data(paths, args)

    elif args.phase == "extract":
        for s in strategies:
            phase_extract(paths, args, s)

    elif args.phase == "probe":
        for s in strategies:
            phase_probe(paths, args, s)

    elif args.phase == "visualize":
        for s in strategies:
            phase_visualize(paths, s)

    elif args.phase == "token_pos":
        phase_token_pos(paths, args)

    elif args.phase == "extract_tokenwise":
        phase_extract_tokenwise(paths, args)
        
    elif args.phase == "probe_direction":
        for s in strategies:
            phase_probe_direction(paths, args, s)
    
    elif args.phase == "pca":
        for s in strategies:
            phase_pca(paths, args, s)
    
    elif args.phase == "cka":
        for s in strategies:
            phase_cka(paths, args, s)
    
    elif args.phase == "attention":
        phase_attention(paths, args)
    
    elif args.phase == "analyze_all":
        phase_analyze_all(paths, args, strategies)

    elif args.phase == "probe_sampled":
        for s in strategies:
            for sampling_scope in sampling_scopes:
                phase_probe_sampled(paths, args, s, sampling_scope)

    elif args.phase == "sequence_model":
        for s in strategies:
            for sampling_scope in sampling_scopes:
                for model_type in sequence_models:
                    phase_sequence_model(paths, args, s, model_type, sampling_scope)

    elif args.phase == "summary":
        phase_summary(paths)

    elif args.phase == "case13_all":
        phase_case13_all(paths, args, strategies)


if __name__ == "__main__":
    main()
