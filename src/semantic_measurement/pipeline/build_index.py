#!/usr/bin/env python3
"""
FAISS index construction pipeline step.

Provides two entry points:
- build_faiss_index(): importable function API
- main(): CLI interface
"""

import argparse
import logging
from pathlib import Path

from semantic_measurement.config import DATA_ROOT
from semantic_measurement.index.build_faiss import build_index_from_embeddings


def build_faiss_index(index: str):
    logger = logging.getLogger(__name__)

    embeddings_dir = DATA_ROOT / "embeddings" / f"{index}_mpnet"
    index_output_dir = DATA_ROOT / "indexes" / index

    chunks_index_file = embeddings_dir / "chunks_index.json"
    snippets_file = embeddings_dir / "snippets.json"

    # -------------------------
    # Validate all required files
    # -------------------------
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    if not chunks_index_file.exists():
        raise FileNotFoundError(f"Missing chunks_index.json at: {chunks_index_file}")

    if not snippets_file.exists():
        raise FileNotFoundError(f"Missing snippets.json at: {snippets_file}")

    index_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building FAISS index for {index}")
    logger.info(f"chunks_index: {chunks_index_file}")
    logger.info(f"snippets file: {snippets_file}")
    logger.info(f"Saving FAISS index to: {index_output_dir}")

    build_index_from_embeddings(
        chunks_index_file=chunks_index_file,
        snippets_file=snippets_file,
        output_dir=index_output_dir,
    )

    logger.info(f"Finished building FAISS index for {index}.")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for a market index.")
    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Market index to process."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_faiss_index(args.index)


if __name__ == "__main__":
    main()
