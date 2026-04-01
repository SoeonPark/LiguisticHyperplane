"""
main.py

Orchestrator for the Linguistic Hyperplane verification pipeline.

Research Questions:
  R1: Does a Linguistic Hyperplane exist?
      (Can hidden states linearly separate hallucination vs non-hallucination?)

Phases:
  data        → Phase 0: Build Case 1 / Case 3 dataset from HotpotQA
  extract     → Phase 1: Extract per-layer hidden states
  probe       → Phase 2: Train linear probes, compute layer-wise AUROC
  visualize   → Phase 3: t-SNE plots, layer accuracy curves
  token_pos   → Phase 4: Compare first / mean / last token positions
  all         → Run all phases in order

Usage:
  python main.py --phase all
  python main.py --phase data
  python main.py --phase probe --strategy first
  python main.py --phase token_pos
"""

import argparse
import os

import config
import data_utils as du
import extract_hidden_state as ehs
import layer_analysis as la
import linear_probe as lp


# ── Phase 0: Data preparation ─────────────────────────────────────────────────

def phase_data() -> None:
    print("\n" + "=" * 60)
    print("Phase 0: HotpotQA case filtering")
    print("=" * 60)

    if os.path.exists(config.CASE_DATA_PATH):
        print(f"[skip] Case file already exists: {config.CASE_DATA_PATH}")
        print("       Delete the file to re-run filtering.")
        return

    model, tokenizer = du.load_model_and_tokenizer()
    cases = du.run_case_filtering(model, tokenizer, max_samples=config.MAX_SAMPLES)
    du.save_cases(cases, config.CASE_DATA_PATH)


# ── Phase 1: Hidden state extraction ─────────────────────────────────────────

def phase_extract(strategy: str = "first") -> None:
    print("\n" + "=" * 60)
    print(f"Phase 1: Hidden state extraction  [strategy={strategy}]")
    print("=" * 60)

    hs_path = os.path.join(config.HIDDEN_STATE_DIR, f"hs_{strategy}.npy")
    if os.path.exists(hs_path):
        print(f"[skip] Hidden states already exist: {hs_path}")
        return

    cases            = du.load_cases(config.CASE_DATA_PATH)
    model, tokenizer = du.load_model_and_tokenizer()

    hidden_states, labels = ehs.extract_all_hidden_states(
        model, tokenizer, cases, strategy=strategy
    )
    ehs.save_hidden_states(hidden_states, labels, strategy=strategy)


# ── Phase 2: Linear probe training ───────────────────────────────────────────

def phase_probe(strategy: str = "first") -> None:
    print("\n" + "=" * 60)
    print(f"Phase 2: Linear probe training  [strategy={strategy}]")
    print("=" * 60)

    result_path = os.path.join(config.PROBE_RESULT_DIR, f"probe_{strategy}.json")
    if os.path.exists(result_path):
        print(f"[skip] Probe results already exist: {result_path}")
        results = lp.load_probe_results(strategy)
        lp.print_summary(results)
        return

    hidden_states, labels = ehs.load_hidden_states(strategy)
    results = lp.train_probe_per_layer(hidden_states, labels)
    lp.save_probe_results(results, strategy)
    lp.print_summary(results)


# ── Phase 3: Visualization ────────────────────────────────────────────────────

def phase_visualize(strategy: str = "first") -> None:
    print("\n" + "=" * 60)
    print(f"Phase 3: Visualization  [strategy={strategy}]")
    print("=" * 60)

    results       = lp.load_probe_results(strategy)
    hidden_states, labels = ehs.load_hidden_states(strategy)

    # Layer-wise accuracy / AUROC curve
    la.plot_layer_accuracy({strategy: results})

    # t-SNE at key layers (early / mid / late / final)
    la.plot_tsne(hidden_states, labels, strategy=strategy)


# ── Phase 4: Token position comparison ───────────────────────────────────────

def phase_token_pos() -> None:
    """
    Compare layer-wise AUROC for first / mean / last token positions.
    Determines which hidden state position is most discriminative for
    the Linguistic Hyperplane.
    """
    print("\n" + "=" * 60)
    print("Phase 4: Token position comparison (first / mean / last)")
    print("=" * 60)

    strategies    = ["first", "mean", "last"]
    results_dict  = {}

    cases = du.load_cases(config.CASE_DATA_PATH)

    for strategy in strategies:
        hs_path = os.path.join(config.HIDDEN_STATE_DIR, f"hs_{strategy}.npy")

        # Extract if not cached
        if not os.path.exists(hs_path):
            print(f"\nExtracting hidden states for strategy='{strategy}'...")
            model, tokenizer = du.load_model_and_tokenizer()
            hidden_states, labels = ehs.extract_all_hidden_states(
                model, tokenizer, cases, strategy=strategy
            )
            ehs.save_hidden_states(hidden_states, labels, strategy=strategy)

        # Train probe if not cached
        result_path = os.path.join(config.PROBE_RESULT_DIR, f"probe_{strategy}.json")
        if not os.path.exists(result_path):
            hidden_states, labels = ehs.load_hidden_states(strategy)
            results = lp.train_probe_per_layer(hidden_states, labels)
            lp.save_probe_results(results, strategy)
        else:
            results = lp.load_probe_results(strategy)

        results_dict[strategy] = results
        lp.print_summary(results)

    # Overlay comparison plot
    la.plot_token_position_comparison(results_dict)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linguistic Hyperplane Verification Pipeline"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "data", "extract", "probe", "visualize", "token_pos"],
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="first",
        choices=["first", "mean", "last", "all"],
        help="Token pooling strategy for hidden state extraction.",
    )
    args = parser.parse_args()

    # Create all output directories
    for d in [
        config.OUTPUT_DIR,
        config.LOG_DIR,
        config.HIDDEN_STATE_DIR,
        config.PROBE_RESULT_DIR,
        config.FIGURE_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    if args.phase == "all":
        phase_data()
        phase_extract(args.strategy)
        phase_probe(args.strategy)
        phase_visualize(args.strategy)
        phase_token_pos()

    elif args.phase == "data":
        phase_data()

    elif args.phase == "extract":
        phase_extract(args.strategy)

    elif args.phase == "probe":
        phase_probe(args.strategy)

    elif args.phase == "visualize":
        phase_visualize(args.strategy)

    elif args.phase == "token_pos":
        phase_token_pos()


if __name__ == "__main__":
    main()