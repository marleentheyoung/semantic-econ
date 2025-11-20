"""
Threshold calibration for semantic similarity.

Evaluates metrics across a threshold range (e.g., 0.40–0.60),
computes precision, recall, FPR, F1 for each threshold, and
selects the optimal threshold (default = max F1).
"""

from typing import List, Dict
import numpy as np


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------

def compute_metrics_for_threshold(
    scores: List[float],
    labels: List[int],
    threshold: float
) -> Dict:
    """Compute TP, FP, FN, TN, precision, recall, FPR, F1 for τ."""
    predictions = [1 if s >= threshold else 0 for s in scores]

    TP = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    FP = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    FN = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)
    TN = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    print({
        "threshold": threshold,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    })
    return {
        "threshold": threshold,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


# ----------------------------------------------------------------------
# Sweep thresholds
# ----------------------------------------------------------------------

def sweep_thresholds(
    scores: List[float],
    labels: List[int],
    start: float = 0.40,
    end: float = 0.60,
    step: float = 0.02
) -> List[Dict]:
    """Evaluate metrics for τ in [start, end] with step size."""
    print(start, end, step)
    thresholds = np.arange(start, end + 1e-9, step)

    print(thresholds)
    return [
        compute_metrics_for_threshold(scores, labels, τ)
        for τ in thresholds
    ]


# ----------------------------------------------------------------------
# Select optimal threshold
# ----------------------------------------------------------------------

def pick_best_threshold(
    results: List[Dict],
    strategy: str = "f1"
) -> Dict:
    """
    Pick the threshold using a strategy:
        - "f1": maximize F1
        - "youden": maximize (TPR - FPR)
    """
    if strategy == "f1":
        return max(results, key=lambda r: r["f1"])

    if strategy == "youden":
        return max(results, key=lambda r: r["recall"] - r["fpr"])

    raise ValueError(f"Unknown strategy: {strategy}")
