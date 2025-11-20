"""
Expert annotation utilities for threshold calibration.

Provides:
    - export_for_manual_annotation
    - load_manual_annotations
"""

from typing import List, Dict, Any
import pandas as pd


def export_for_manual_annotation(
    candidates: List[Dict[str, Any]],
    output_path: str
):
    """
    Export Excel file for manual annotation.

    Columns:
        pattern, call_id, section, similarity, text, relevant
    """
    df = pd.DataFrame({
        "pattern": [c["pattern"] for c in candidates],
        "call_id": [c["call_id"] for c in candidates],
        "section": [c["section"] for c in candidates],
        "similarity": [c["similarity"] for c in candidates],
        "text": [c["text"] for c in candidates],
        "relevant": [None] * len(candidates),  # empty column for annotation
    })

    df.to_excel(output_path, index=False)


def load_manual_annotations(
    input_path: str
) -> List[Dict[str, Any]]:
    """
    Load manually annotated file and return labeled candidates.

    Input file must contain:
        pattern, call_id, section, similarity, text, relevant
    where `relevant` is 0 or 1.
    """
    df = pd.read_excel(input_path)

    results = []
    for _, row in df.iterrows():
        results.append({
            "pattern": row["pattern"],
            "call_id": row["call_id"],
            "section": row["section"],
            "similarity": float(row["similarity"]),
            "text": row["text"],
            "label": int(row["relevant"]),
        })

    return results
