#!/usr/bin/env python3
"""
Call metadata builder pipeline step.

Provides:
- build_call_metadata_for_index(): importable wrapper
- main(): CLI interface
"""

import argparse
import logging
from pathlib import Path

from semantic_measurement.config.global_calibration import DATA_ROOT
from semantic_measurement.data.call_metadata import build_call_metadata


def build_call_metadata_for_index(index: str):
    """
    Wrapper that prepares paths and calls the actual build_call_metadata().

    Parameters
    ----------
    index : str
        "SP500" or "STOXX600"
    """
    logger = logging.getLogger(__name__)

    # Input: FAISS-related embedding / index directory
    embedding_dir = DATA_ROOT / "indexes" / index

    # Output: central metadata folder
    metadata_output_dir = DATA_ROOT / "metadata"
    metadata_output_dir.mkdir(parents=True, exist_ok=True)

    # Output file: SP500_calls.json or STOXX600_calls.json
    output_file = metadata_output_dir / f"{index}_calls.json"

    logger.info(f"Building call metadata for index {index}")
    logger.info(f"Using embedding/index dir: {embedding_dir}")
    logger.info(f"Writing metadata to:       {output_file}")

    if not embedding_dir.exists():
        raise FileNotFoundError(f"Index folder does not exist: {embedding_dir}")

    # Call the real metadata builder with correct args
    build_call_metadata(
        embedding_dir=embedding_dir,
        stock_index=index,
        output_file=output_file
    )

    logger.info(f"Finished building call metadata for {index}.")


def main():
    parser = argparse.ArgumentParser(description="Build call metadata for a market index.")

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Market index to process."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_call_metadata_for_index(args.index)


if __name__ == "__main__":
    main()
