#!/usr/bin/env python3
"""
Run semantic retrieval for a given concept/topic across SP500 + STOXX600.

Usage:
    python scripts/run_retrieval_pipeline.py --topic climate_risk --top-k 200
"""

import argparse
import json
from pathlib import Path
import logging

from semantic_measurement.config import DATA_ROOT
from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.embeddings.sentence_transformers_backend import (
    SentenceTransformerBackend,
)


def load_queries(topic_file: Path):
    if not topic_file.exists():
        raise FileNotFoundError(f"Query file not found: {topic_file}")

    with topic_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic retrieval for a given concept."
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Topic name (expects data/queries/<topic>.json)",
    )

    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")

    args = parser.parse_args()
    topic = args.topic

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---------------------------------------------------------
    # Resolve paths
    # ---------------------------------------------------------
    queries_file = DATA_ROOT / "queries" / f"{topic}.json"
    retrieval_out_dir = DATA_ROOT / "retrieval" / topic
    retrieval_out_dir.mkdir(parents=True, exist_ok=True)

    output_file = retrieval_out_dir / f"{topic}_raw_retrieval.jsonl"

    logger.info(f"Loading concept queries: {queries_file}")
    queries = load_queries(queries_file)

    # ---------------------------------------------------------
    # Init embedder & retriever (SP500 + STOXX600)
    # ---------------------------------------------------------
    embedder = SentenceTransformerBackend(model_name=args.model)
    retriever = SemanticRetriever(
        embedder=embedder,
        indexes_dir=DATA_ROOT / "indexes",
        index_names=["SP500", "STOXX600"],
    )

    logger.info("Running retrieval...")
    logger.info(f"Top-k per query: {args.top_k}")

    # ---------------------------------------------------------
    # For each query → embed → retrieve → write
    # ---------------------------------------------------------
    with output_file.open("w", encoding="utf-8") as out_f:
        for q in queries:
            q_text = q["pattern"]
            q_id = q.get("id", q_text[:30])

            logger.info(f"Query: {q_id}")

            q_emb = embedder.encode([q_text])[0]

            # unified top-k across US + EU
            results = retriever.search_by_embedding(q_emb, top_k=args.top_k)

            for r in results:
                out_f.write(
                    json.dumps(
                        {
                            "query_id": q_id,
                            "query_text": q_text,
                            "similarity": float(r["score"]),
                            "faiss_id": r["faiss_id"],
                            "index": r["index_name"],
                            "paragraph": r["snippet"]["text"],
                            "paragraph_meta": r["snippet"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    logger.info(f"Finished. Wrote retrieval results to: {output_file}")


if __name__ == "__main__":
    main()
