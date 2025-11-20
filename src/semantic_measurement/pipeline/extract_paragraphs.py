#!/usr/bin/env python

"""
Paragraph extraction pipeline step.

Provides two entry points:
- extract_paragraphs(): importable function API
- main(): CLI interface
"""

import argparse
import logging
from pathlib import Path

from semantic_measurement.config.global_calibration import DATA_ROOT
from semantic_measurement.preprocessing.paragraph_extractor import (
    extract_paragraphs_from_folder,
)


def extract_paragraphs(index: str):
    """
    Importable function to extract paragraph-level records
    from structured transcripts.

    Parameters
    ----------
    index : str
        Either "SP500" or "STOXX600".
    """
    logger = logging.getLogger(__name__)

    data_root: Path = DATA_ROOT

    # -------------------------------------------
    # Resolve paths using DATA_ROOT + index
    # -------------------------------------------
    input_folder = data_root / "intermediaries" / "structured_calls" / index
    output_folder = data_root / "intermediaries" / "paragraphs"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{index}_paragraphs.jsonl"

    logger.info("Index: %s", index)
    logger.info("DATA_ROOT: %s", data_root)
    logger.info("Input (structured_calls): %s", input_folder)
    logger.info("Output (paragraphs JSONL): %s", output_file)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # -------------------------------------------
    # Execute paragraph extraction
    # -------------------------------------------
    extract_paragraphs_from_folder(
        input_folder=str(input_folder),
        output_file=str(output_file),
    )

    logger.info("Finished extracting paragraphs for index %s.", index)


def main():
    parser = argparse.ArgumentParser(
        description="Extract paragraph-level records from structured transcripts."
    )

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Market index to process.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    extract_paragraphs(args.index)


if __name__ == "__main__":
    main()
