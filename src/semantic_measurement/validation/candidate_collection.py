"""
Candidate collection for threshold calibration.

This module now supports STRATIFIED sampling across similarity bins
to avoid top-k bias (where all retrieved paragraphs are too similar).
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np


def deduplicate_candidates(raw_candidates):
    """
    Deduplicate retrieval results across multiple queries.
    Uses (call_id, faiss_id) as unique paragraph key.

    Keeps ONLY the candidate with the highest similarity score.
    """
    dedup = {}

    for c in raw_candidates:
        # Define the unique identity of a snippet
        key = (c["call_id"], c["faiss_id"])

        # Insert if new
        if key not in dedup:
            dedup[key] = c
        else:
            # Keep the highest similarity
            if c["similarity"] > dedup[key]["similarity"]:
                dedup[key] = c

    return list(dedup.values())

# ----------------------------------------------------------------------
# 1. Raw retrieval (no threshold, no stratification)
# ----------------------------------------------------------------------

def collect_candidates_raw(
    topic: str,
    queries: List[str],
    retriever,
    top_k: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Retrieve a large pool of raw candidates for a topic.

    This is used internally for stratified sampling.
    """
    items = []

    for q in queries:
        results = retriever.search_by_text(q, top_k=top_k)

        for r in results:
            snippet = r["snippet"]

            items.append({
                "topic": topic,
                "pattern": q,
                "text": snippet["text"],
                "similarity": r["score"],
                "call_id": snippet["call_id"],
                "section": snippet.get("section"),
                "faiss_id": r["faiss_id"],
                "index_name": r["index_name"],
            })

    return items


# ----------------------------------------------------------------------
# 2. Stratified sampling candidates
# ----------------------------------------------------------------------

def collect_candidates_stratified(
    topic: str,
    queries: List[str],
    retriever,
    top_k_raw: int = 1000,
    bins: List[Tuple[float, float]] = None,
    samples_per_bin: int = 25,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Stratified candidate sampling for threshold calibration.

    Retrieves many results (top_k_raw), bins them by similarity,
    and samples evenly across bins.
    """

    # Default bins capturing the threshold zone
    if bins is None:
        bins = [
            (0.80, 1.00),
            (0.70, 0.80),
            (0.60, 0.70),
            (0.50, 0.60),
            (0.40, 0.50),
            (0.30, 0.40),
        ]

    # Step 1: retrieve raw candidates
    raw = collect_candidates_raw(topic, queries, retriever, top_k=top_k_raw)

    raw = deduplicate_candidates(raw)
    raw = sorted(raw, key=lambda x: x["similarity"], reverse=True)

    # Step 2: allocate into bins
    binned = {i: [] for i in range(len(bins))}
    for c in raw:
        sim = c["similarity"]
        for i, (low, high) in enumerate(bins):
            if low <= sim < high:
                binned[i].append(c)
                break

    # 🔥 Verbose: print bin distribution
    if verbose:
        print("\nRaw candidates:", len(raw))
        print("Bin distribution:")
        for i, (low, high) in enumerate(bins):
            print(f"  {low:.2f}–{high:.2f}: {len(binned[i])} items")

    # Step 3: stratified sampling
    rng = np.random.default_rng()
    final = []

    for i, (low, high) in enumerate(bins):
        candidates = binned[i]
        if len(candidates) == 0:
            continue

        n = min(samples_per_bin, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        final.extend([candidates[j] for j in idx])

    return final


# ----------------------------------------------------------------------
# 3. Unified interface
# ----------------------------------------------------------------------

def collect_candidates(
    topic: str,
    queries: List[str],
    retriever,
    stratified: bool = True,
    top_k_raw: int = 1000,
    samples_per_bin: int = 25,
    bins: List[Tuple[float, float]] = None,
    verbose: bool = False,
):
    
    if stratified:
        return collect_candidates_stratified(
            topic=topic,
            queries=queries,
            retriever=retriever,
            top_k_raw=top_k_raw,
            samples_per_bin=samples_per_bin,
            bins=bins,
            verbose=verbose,      # <-- pass it through
        )

    # fallback: simple top-k (old mode)
    return collect_candidates_raw(topic, queries, retriever, top_k=top_k_raw)
