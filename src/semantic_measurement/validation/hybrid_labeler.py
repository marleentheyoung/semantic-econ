# src/semantic_measurement/validation/hybrid_labeler.py

"""
Hybrid human–LLM labeling module.

Process:
    1. LLM labels candidates
    2. User interactively verifies selected items
    3. Inter-annotator agreement is computed
"""

import sys
import termios
import tty
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import cohen_kappa_score


# ----------------------------------------------------------
# Utilities: single character input (non-blocking echo)
# ----------------------------------------------------------

def _get_single_char() -> str:
    """Wait for a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ----------------------------------------------------------
# Interactive CLI annotation
# ----------------------------------------------------------

def interactive_review(labeled_llm: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Interactively verify LLM labels. User can stop at any time.

    Keys:
        y = relevant
        n = not relevant
        s = skip
        q = quit
    """

    print("\nStarting interactive human verification...")
    print("Keys: [y] yes  [n] no  [s] skip  [q] quit\n")

    annotated = []

    for idx, item in enumerate(labeled_llm, start=1):
        text = item["text"]
        llm_label = item["label"]

        print("\n" + "-" * 60)
        print(f"Snippet #{idx}   similarity={item['similarity']:.3f}")
        print(f"LLM says: {'YES' if llm_label == 1 else 'NO'}")
        print("-" * 60)
        print(text)
        print("\nYour label? [y/n/s/q] ", end="", flush=True)

        key = _get_single_char().lower()
        print(key)

        if key == "q":
            print("\nStopping early per user request.")
            break
        elif key == "y":
            human_label = 1
        elif key == "n":
            human_label = 0
        elif key == "s":
            human_label = None
        else:
            print("Invalid key, skipping.")
            human_label = None

        annotated.append({
            **item,
            "label_human": human_label,
        })

    return annotated


# ----------------------------------------------------------
# Agreement computation
# ----------------------------------------------------------

def compute_agreement(annotated: List[Dict[str, Any]]):
    """Compute inter-annotator agreement between human and LLM."""

    human_labels = []
    llm_labels = []

    for item in annotated:
        if item.get("label_human") is None:
            continue
        human_labels.append(item["label_human"])
        llm_labels.append(item["label"])

    if len(human_labels) == 0:
        print("No human-labeled items — agreement cannot be computed.")
        return None

    human_labels = np.array(human_labels)
    llm_labels = np.array(llm_labels)

    percent = (human_labels == llm_labels).mean()
    kappa = cohen_kappa_score(human_labels, llm_labels)

    # Confusion matrix
    TP = int(((human_labels == 1) & (llm_labels == 1)).sum())
    TN = int(((human_labels == 0) & (llm_labels == 0)).sum())
    FP = int(((human_labels == 0) & (llm_labels == 1)).sum())
    FN = int(((human_labels == 1) & (llm_labels == 0)).sum())

    print("\n=== Human–LLM Agreement Report ===")
    print(f"Human-labeled samples: {len(human_labels)}")
    print(f"Percent agreement: {percent*100:.1f}%")
    print(f"Cohen’s κ: {kappa:.3f}")
    print("\nConfusion matrix (LLM vs Human):")
    print(f"  TP: {TP}   FP: {FP}")
    print(f"  FN: {FN}   TN: {TN}")
    print("------------------------------------\n")

    return {
        "percent": percent,
        "kappa": kappa,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }


# ----------------------------------------------------------
# High-level wrapper
# ----------------------------------------------------------

def stratified_random_sample(
    items: List[Dict[str, Any]],
    bins=None,
    samples_per_bin=15,
) -> List[Dict[str, Any]]:
    """
    Stratify items by similarity and then randomly sample from each bin.
    """
    if bins is None:
        bins = [
            (0.80, 1.00),
            (0.70, 0.80),
            (0.60, 0.70),
            (0.50, 0.60),
            (0.40, 0.50),
            (0.30, 0.40),
        ]

    # allocate bins
    binned = {i: [] for i in range(len(bins))}
    for item in items:
        sim = item["similarity"]
        for i, (low, high) in enumerate(bins):
            if low <= sim < high:
                binned[i].append(item)
                break

    # sample equally
    rng = np.random.default_rng()
    selected = []
    for i, group in binned.items():
        if len(group) == 0:
            continue
        n = min(samples_per_bin, len(group))
        idx = rng.choice(len(group), size=n, replace=False)
        selected.extend([group[j] for j in idx])

    # shuffle final combined list
    rng.shuffle(selected)
    return selected


def run_hybrid_labeling(labeled_llm: List[Dict[str, Any]]):
    """
    Full hybrid labeling procedure.
    - Performs stratified + randomized sample selection
    - Runs interactive review
    - Computes agreement
    """
    print("\nPreparing stratified & randomized sample for human review...")

    # stratify & shuffle
    sample = stratified_random_sample(labeled_llm)

    print(f"Selected {len(sample)} items for hybrid annotation.\n")

    annotated = interactive_review(sample)
    compute_agreement(annotated)
    return annotated
