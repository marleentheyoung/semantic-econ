#!/usr/bin/env python3
"""
Embedding pipeline step.

Provides two entry points:
- embed_segments(): importable function API
- main(): CLI interface

This step now produces:
- chunks/*.npy                 (embedding shards)
- chunks_index.json            (list of chunk paths)
- snippets.parquet             (FAISS-aligned paragraph metadata)
- embedding_metadata.json      (summary)
"""

import argparse
import json
from pathlib import Path
import logging

from semantic_measurement.embeddings.embed_segments import embed_paragraphs_to_chunks
from semantic_measurement.data.schemas import ParagraphRecord
from semantic_measurement.config.global_calibration import DATA_ROOT, DEFAULT_EMBEDDING_MODEL


def load_paragraphs(path: Path):
    """Load paragraph records from .json or .jsonl."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [ParagraphRecord(**json.loads(l)) for l in f if l.strip()]
    else:
        with path.open("r", encoding="utf-8") as f:
            arr = json.load(f)
        return [ParagraphRecord(**d) for d in arr]


def embed_segments(
    index: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 150_000,
    batch_size: int = 64,
):
    """
    Compute chunked embeddings + metadata parquet for one index.

    Produces:
        embeddings/{index}_mpnet/
            - chunks/
            - chunks_index.json
            - snippets.parquet
            - embedding_metadata.json
    """
    logger = logging.getLogger(__name__)

    paragraphs_file = (
        DATA_ROOT /
        "intermediaries" /
        "paragraphs" /
        f"{index}_paragraphs.jsonl"
    )

    output_dir = DATA_ROOT / "embeddings" / f"{index}_mpnet"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading paragraphs from: {paragraphs_file}")
    logger.info(f"Writing embeddings + metadata to: {output_dir}")

    if not paragraphs_file.exists():
        raise FileNotFoundError(f"Paragraphs file does not exist: {paragraphs_file}")

    paragraphs = load_paragraphs(paragraphs_file)

    logger.info(f"Loaded {len(paragraphs)} paragraphs. Beginning embedding...")

    metadata = embed_paragraphs_to_chunks(
        paragraphs=paragraphs,
        output_dir=output_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )

    logger.info("Embedding completed.")
    logger.info(json.dumps(metadata, indent=2))

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Embed paragraphs into MPNet vectors for a given index."
    )

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Market index to process."
    )

    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    embed_segments(
        index=args.index,
        model_name=args.model,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
