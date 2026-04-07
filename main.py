"""
main.py - minimal fix: phase_token_pos 항상 ALL_STRATEGIES 고정
"""

import argparse
import os
import sys

_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg in ("--gpu", "-g") and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import config
import data_utils as du
import extract_hidden_state as ehs
import layer_analysis as la
import linear_probe as lp
import analysis_extended as ae

ALL_STRATEGIES = ["first", "mean", "last"]


def make_model_short(model_name: str) -> str:
    short = model_name.split("/")[-1].lower()
    short = short.replace("meta-llama-", "llama").replace("meta-", "")
    short = short.replace("mistralai-", "").replace("microsoft-", "")
    return short


def build_paths(model_name: str, subset: str, data_split: str, tag: str) -> dict:
    short    = make_model_short(model_name)
    split_s  = "val" if data_split == "validation" else data_split
    exp_name = f"{short}__{subset}_{split_s}"
    if tag:
        exp_name += f"__{tag}"
    base = os.path.join("outputs", exp_name)
    return {
        "base": base,
        "cases": os.path.join(base, "cases.json"),
        "cases_all": os.path.join(base, "cases_all.json"),
        "hs_dir": os.path.join(base, "hidden_states"),
        "probe_dir": os.path.join(base, "probe_results"),
        "figure_dir": os.path.join(base, "figures"),
        "log_dir": os.path.join(base, "logs"),
        "exp_name": exp_name,
    }


def _strategies_to_run(strategy_arg: str):
    """--strategy all  →  ['first', 'mean', 'last'],  else single."""
    return ALL_STRATEGIES if strategy_arg == "all" else [strategy_arg]


# ── Phase 0 ───────────────────────────────────────────────────────────────────
def phase_data(paths: dict, args) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 0: Data filtering  [{paths['exp_name']}]")
    print(f"  subset={args.subset}  split={args.data_split}  max_samples={args.max_samples or 'all'}")
    print("=" * 60)

    if os.path.exists(paths["cases"]):
        print(f"[skip] Already exists: {paths['cases']}")
        return

    model, tokenizer = du.load_model_and_tokenizer(args.model)
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
    if os.path.exists(hs_path):
        print(f"[skip] Already exists: {hs_path}")
        return

    cases = du.load_cases(paths["cases"])
    model, tokenizer = du.load_model_and_tokenizer(args.model)
    hidden_states, labels = ehs.extract_all_hidden_states(
        model, tokenizer, cases, strategy=strategy
    )
    ehs.save_hidden_states(hidden_states, labels, strategy=strategy, out_dir=paths["hs_dir"])


# ── Phase 2 ───────────────────────────────────────────────────────────────────
def phase_probe(paths: dict, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 2: Linear probe  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)

    result_path = os.path.join(paths["probe_dir"], f"probe_{strategy}.json")
    if os.path.exists(result_path):
        print(f"[skip] Already exists: {result_path}")
        results = lp.load_probe_results(strategy, probe_dir=paths["probe_dir"])
        lp.print_summary(results)
        return

    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    cases = du.load_cases(paths["cases"])
    results = lp.train_probe_per_layer(hidden_states, labels, cases=cases)
    lp.save_probe_results(results, strategy, probe_dir=paths["probe_dir"])
    lp.print_summary(results)


# ── Phase 3 ───────────────────────────────────────────────────────────────────
def phase_visualize(paths: dict, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 3: Visualization  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)

    results = lp.load_probe_results(strategy, probe_dir=paths["probe_dir"])
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])

    la.plot_layer_accuracy({strategy: results}, figure_dir=paths["figure_dir"])
    la.plot_tsne(hidden_states, labels, strategy=strategy, figure_dir=paths["figure_dir"])


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
        cases = du.load_cases(paths["cases"])
        phase_probe(paths, strategy)
        results_dict[strategy] = lp.load_probe_results(strategy, probe_dir=paths["probe_dir"])
        lp.print_summary(results_dict[strategy])

    la.plot_token_position_comparison(results_dict, figure_dir=paths["figure_dir"])


# ── Phase 5: Probe Direction ──────────────────────────────────────────────────
def phase_probe_direction(paths: dict, args, strategy: str) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 5: Probe direction analysis  [{paths['exp_name']}]  strategy={strategy}")
    print("=" * 60)
 
    hidden_states, labels = ehs.load_hidden_states(strategy, hs_dir=paths["hs_dir"])
    model, tokenizer = du.load_model_and_tokenizer(args.model)
 
    ae.analyze_probe_direction(
        hidden_states=hidden_states,
        labels=labels,
        model=model,
        tokenizer=tokenizer,
        probe_dir=paths["probe_dir"],
        strategy=strategy,
        figure_dir=paths["figure_dir"],
    )
 
 
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
 
 
# ── Phase 8: Attention to Context ─────────────────────────────────────────────
def phase_attention(paths: dict, args) -> None:
    print("\n" + "=" * 60)
    print(f"Phase 8: Attention to context  [{paths['exp_name']}]")
    print("=" * 60)
 
    cases = du.load_cases(paths["cases"])
    model, tokenizer = du.load_model_and_tokenizer(args.model)
 
    ae.analyze_attention_to_context(
        model=model,
        tokenizer=tokenizer,
        cases=cases,
        figure_dir=paths["figure_dir"],
        max_samples=args.attn_max_samples,
    )
 
 
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
                            "probe_direction", "pca", "cka", "attention", "analyze_all"
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
    parser.add_argument("--gpu", "-g", type=str, default="0")
    parser.add_argument("--attn_max_samples", type=int, default=200, help="Max samples for attention analysis (Phase 8). Default 200.")

    args = parser.parse_args()
    lp.BALANCED = args.balanced

    paths = build_paths(args.model, args.subset, args.data_split, args.tag)
    for key, val in paths.items():
        if key not in ("cases", "exp_name") and not val.endswith(".json"):
            os.makedirs(val, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Subset     : {args.subset}")
    print(f"  Split      : {args.data_split}  (max_samples={args.max_samples or 'all'})")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Balanced   : {args.balanced}")
    print(f"  Experiment : {paths['exp_name']}")
    print(f"  GPU        : {_gpu}")
    print(f"{'='*60}")

    strategies = _strategies_to_run(args.strategy)

    if args.phase == "all":
        phase_data(paths, args)
        for s in strategies:
            phase_extract(paths, args, s)
            phase_probe(paths, s)
            phase_visualize(paths, s)
        phase_token_pos(paths, args)  

    elif args.phase == "data":
        phase_data(paths, args)

    elif args.phase == "extract":
        for s in strategies:
            phase_extract(paths, args, s)

    elif args.phase == "probe":
        for s in strategies:
            phase_probe(paths, s)

    elif args.phase == "visualize":
        for s in strategies:
            phase_visualize(paths, s)

    elif args.phase == "token_pos":
        phase_token_pos(paths, args)
        
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


if __name__ == "__main__":
    main()