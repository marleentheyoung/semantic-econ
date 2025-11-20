#!/usr/bin/env python3
"""
Run threshold calibration for a topic.

This script orchestrates:
    - Candidate collection
    - LLM-based labeling or expert annotation
    - Threshold sweeping (0.40–0.60)
    - Optimal threshold selection
    - Updated query config saved

"""

import argparse
import json
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from semantic_measurement.validation.candidate_collection import collect_candidates
from semantic_measurement.validation.llm_labeler import llm_label_candidates
from semantic_measurement.validation.expert_annotation import (
    export_for_manual_annotation,
    load_manual_annotations,
)
from semantic_measurement.validation.threshold_calibration import (
    sweep_thresholds,
    pick_best_threshold,
)
from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.validation.hybrid_labeler import run_hybrid_labeling

from semantic_measurement.config.global_calibration import DATA_ROOT
from pathlib import Path

from semantic_measurement.config.calibration import load_calibration_config
from semantic_measurement.utils.helpers import load_queries

CAL_CFG = load_calibration_config()

def default_expert_path(topic: str) -> Path:
    root = DATA_ROOT / CAL_CFG["expert"]["root"]
    fname = CAL_CFG["expert"]["filename"]
    return root / topic / fname

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def update_threshold_in_config(query_file: Path, queries: dict, threshold: float):
    queries["threshold"] = float(threshold)
    with open(query_file, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"\n✓ Saved threshold {threshold:.3f} to {query_file}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Threshold calibration")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--mode", required=True, choices=["llm", "expert", "hybrid"])
    parser.add_argument("--export", nargs="?", const=True, help="Export expert samples")
    parser.add_argument("--import", dest="import_path", nargs="?", const=True,
                            help="Import expert-labeled samples")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--start", type=float, default=CAL_CFG["sweep"]["start"])
    parser.add_argument("--end", type=float, default=CAL_CFG["sweep"]["end"])
    parser.add_argument("--step", type=float, default=CAL_CFG["sweep"]["step"])


    args = parser.parse_args()

    # Load queries
    queries, query_file = load_queries(args.topic)

    # Initialize retriever
    retriever = SemanticRetriever()

    print(f"\n=== Threshold Calibration for topic: {args.topic} ===")
    print("Collecting candidates...")

    candidates = collect_candidates(
        topic=queries.get("topic", args.topic),
        queries=queries["queries"],
        retriever=retriever,
        stratified=True,
        bins=CAL_CFG["retrieval"]["bins"],
        top_k_raw=CAL_CFG["retrieval"]["top_k_raw"],
        samples_per_bin=CAL_CFG["retrieval"]["samples_per_bin"],
        verbose=args.verbose,   
    )


    print(f"✓ Retrieved {len(candidates)} candidates")

    # ------------------------------------------
    # Expert mode
    # ------------------------------------------
    if args.mode == "expert":
        if args.export:
            export_path = default_expert_path(args.topic)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            if export_path.exists():
                print(f"\n⚠️ Expert annotation file already exists:\n   {export_path}")
                confirm = input("Overwrite existing file? [y/N]: ").strip().lower()

                if confirm not in ("y", "yes"):
                    print("Aborting export. No new file created.")
                    return

            # Write new file (overwrite allowed)
            export_for_manual_annotation(candidates, export_path)

            print(f"\n✓ Exported {len(candidates)} samples to {export_path}")
            print("Annotate them manually, then re-run with --import.")
            return

        # -------------------------------------------------
        # Case 2: Expert IMPORT — read annotated file
        # -------------------------------------------------
        if args.import_path:
            import_path = default_expert_path(args.topic)

            if not import_path.exists():
                raise FileNotFoundError(
                    f"Cannot import expert annotations. File not found: {import_path}\n"
                    "Run with --export first."
                )

            labeled = load_manual_annotations(import_path)
            print(f"✓ Loaded {len(labeled)} human-annotated samples")

        else:
            raise ValueError("Expert mode requires either --export or --import")


    # ------------------------------------------
    # LLM mode
    # ------------------------------------------
    elif args.mode == "llm":
        print("\nRunning LLM labeling...")
        labeled = llm_label_candidates(queries['description'], candidates)

    elif args.mode == "hybrid":
        print("\nRunning initial LLM labeling...")
        labeled_llm = llm_label_candidates(queries['description'], candidates)
        labeled = run_hybrid_labeling(labeled_llm)

    # Extract score + label vectors
    scores = [c["similarity"] for c in labeled]
    labels = [c["label"] for c in labeled]

    # Threshold sweep
    print("\nSweeping thresholds...")
    results = sweep_thresholds(scores, labels, args.start, args.end, args.step)

    # Pick best τ
    best = pick_best_threshold(results, strategy="f1")

    print("\n=== Optimal Threshold ===")
    print(f"τ = {best['threshold']:.3f}")
    print(f"Precision = {best['precision']:.3f}")
    print(f"Recall    = {best['recall']:.3f}")
    print(f"F1        = {best['f1']:.3f}")
    print(f"FPR       = {best['fpr']:.3f}")
    print(f"TP/FP/FN/TN: {best['TP']}/{best['FP']}/{best['FN']}/{best['TN']}")

    # Save new threshold
    update_threshold_in_config(query_file, queries, best["threshold"])


if __name__ == "__main__":
    main()
