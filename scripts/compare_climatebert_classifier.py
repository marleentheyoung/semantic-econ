#!/usr/bin/env python3
"""
compare_climatebert_classifier.py

Compare:
1. Your semantic retrieval system (FAISS + calibrated threshold)
2. ClimateBERT classifier (climatebert/distilroberta-base-climate-detector)

Output:
- ClimateBERT probabilities
- Semantic-retriever hits
- Overlap statistics
- CSV + JSON with per-paragraph evaluation
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

# --- your project imports ---
from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.concepts.concept_retriever import ConceptRetriever
from semantic_measurement.config.global_calibration import (
    DATA_ROOT, DEFAULT_INDEX_DIR, DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_NAMES
)
from semantic_measurement.utils.helpers import load_queries
from semantic_measurement.pipeline.call_metadata_loader import load_call_metadata
from semantic_measurement.pipeline.topic_runner import TopicRunner


# =====================================================================
# 1. CLIMATEBERT CLASSIFIER LOADER
# =====================================================================

def load_climatebert_classifier():
    model_name = "climatebert/distilroberta-base-climate-detector"
    print(f"Loading ClimateBERT classifier: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        padding=True,
        max_length=512,
        device=-1,
    )
    return clf


# =====================================================================
# 2. LOAD SNIPPETS WITH A GLOBAL PARAGRAPH ID
# =====================================================================

def load_all_snippets() -> List[Dict[str, Any]]:
    """
    Load snippets.parquet for both SP500 and STOXX600 and assign a global_id.
    This ensures both ClimateBERT and semantic retrieval refer to same universe.
    """
    all_records = []
    current_id = 0

    for name in DEFAULT_INDEX_NAMES:
        path = DATA_ROOT / "embeddings" / f"{name}_mpnet" / "snippets.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

        df = pd.read_parquet(path)
        df["index_name"] = name
        df["faiss_id_in_index"] = df["faiss_id"]

        # Assign global ids
        n = len(df)
        df["global_id"] = range(current_id, current_id + n)
        current_id += n

        all_records.extend(df.to_dict(orient="records"))

    print(f"Loaded total paragraphs: {len(all_records)}")
    return all_records


# =====================================================================
# 3. CLIMATEBERT BATCH CLASSIFICATION
# =====================================================================

def extract_climate_prob(out: List[Dict]) -> float:
    for item in out:
        if item["label"].lower() == "yes":
            return float(item["score"])
    return 0.0


def classify_climatebert(clf, paragraphs, batch_size=32):
    ds = Dataset.from_list(paragraphs).select_columns(["text"])

    print(f"Running ClimateBERT over {len(paragraphs)} paragraphs...")

    probs = []
    iterator = clf(
        KeyDataset(ds, "text"),
        batch_size=batch_size,
        truncation=True,
        padding=True,
    )

    for out in tqdm(iterator, total=len(ds), desc="ClimateBERT"):
        probs.append(extract_climate_prob(out))

    return probs


# =====================================================================
# 4. OVERLAP METRICS
# =====================================================================

def compute_overlap(setA: set, setB: set) -> Dict[str, float]:
    inter = setA & setB
    union = setA | setB
    return {
        "jaccard": len(inter) / len(union) if union else 0.0,
        "intersection": len(inter),
        "A_only": len(setA - setB),
        "B_only": len(setB - setA),
    }


# =====================================================================
# 5. MAIN PIPELINE
# =====================================================================

def main():

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    TOPIC_NAME = "climate"
    CLIMATE_THRESHOLD = 0.5

    print("Loading semantic concept queries...")
    queries, _ = load_queries(TOPIC_NAME)
    retrieval_threshold = queries["threshold"]

    # ---------------------------------------------------------
    # Load snippets (global paragraph universe)
    # ---------------------------------------------------------
    print("Loading paragraph universe...")
    paragraphs = load_all_snippets()

    # Map global_id → snippet
    paragraph_by_gid = {p["global_id"]: p for p in paragraphs}

    # ---------------------------------------------------------
    # Run semantic retrieval
    # ---------------------------------------------------------
    print("Running semantic retrieval...")
    call_metadata = load_call_metadata(DATA_ROOT / "metadata" / "all_calls.json")

    runner = TopicRunner(
        call_metadata=call_metadata,
        queries_root=Path("data/queries"),
        index_dir=DEFAULT_INDEX_DIR,
        embeddings_dir=DEFAULT_EMBEDDINGS_DIR,
        model_name=DEFAULT_EMBEDDING_MODEL,
    )

    hits_by_call = runner.run_topic(TOPIC_NAME)

    # Flatten semantic hits into a global-id set
    semantic_ids = set()
    for call_id, hitlist in hits_by_call.items():
        for h in hitlist:
            snippet = paragraph_by_gid[h.faiss_id]  # global id == faiss_id in parquet system
            semantic_ids.add(snippet["global_id"])

    print(f"Semantic retriever hits ≥ {retrieval_threshold}: {len(semantic_ids)}")

    # ---------------------------------------------------------
    # Run ClimateBERT classifier
    # ---------------------------------------------------------
    clf = load_climatebert_classifier()
    probs = classify_climatebert(clf, paragraphs)

    climate_ids = {
        p["global_id"]
        for p, prob in zip(paragraphs, probs)
        if prob >= CLIMATE_THRESHOLD
    }

    print(f"ClimateBERT hits ≥ {CLIMATE_THRESHOLD}: {len(climate_ids)}")

    # ---------------------------------------------------------
    # Compute Overlap
    # ---------------------------------------------------------
    overlap = compute_overlap(semantic_ids, climate_ids)

    print("\n=== OVERLAP SUMMARY ===")
    for k, v in overlap.items():
        print(f"{k}: {v}")

    # ---------------------------------------------------------
    # Build detailed output table
    # ---------------------------------------------------------
    rows = []
    for p, prob in zip(paragraphs, probs):
        gid = p["global_id"]
        rows.append({
            "global_id": gid,
            "text": p["text"],
            "semantic_hit": int(gid in semantic_ids),
            "semantic_sim": None,  # optional: collect from retrieval if needed
            "climatebert_prob": prob,
            "climatebert_hit": int(gid in climate_ids),
            "faiss_id_in_index": p["faiss_id_in_index"],
            "index_name": p["index_name"],
            "call_id": p.get("call_id"),
        })

    # Save
    out_csv = Path("comparison_classifier_results.csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    out_json = Path("comparison_classifier_results.json")
    with out_json.open("w") as f:
        json.dump({
            "summary": overlap,
            "retrieval_threshold": retrieval_threshold,
            "classifier_threshold": CLIMATE_THRESHOLD,
            "n_paragraphs": len(paragraphs)
        }, f, indent=2)

    print(f"\nSaved CSV → {out_csv}")
    print(f"Saved JSON → {out_json}")


if __name__ == "__main__":
    main()
