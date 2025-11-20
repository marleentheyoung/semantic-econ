#!/usr/bin/env python

"""
Transcript structuring pipeline step.

Provides two entry points:
- structure_transcripts(): importable function API
- main(): CLI interface
"""

import argparse
import logging
from pathlib import Path

from semantic_measurement.config.global_calibration import DATA_ROOT
from semantic_measurement.data.pdf_metadata import build_metadata_lookup
from semantic_measurement.data.structure_transcripts import (
    structure_all_transcripts_from_parts,
)

from semantic_measurement.parsing.qna_parser import parse_qna_section


def structure_transcripts(index: str):
    """
    Importable function to convert raw transcript parts into structured call JSONs.

    Parameters
    ----------
    index : str
        "SP500" or "STOXX600"
    """
    logger = logging.getLogger(__name__)

    data_root: Path = DATA_ROOT

    # -------------------------------------------------------------------------
    # Infer paths from index
    # -------------------------------------------------------------------------
    pdf_root = data_root / "raw" / index
    input_folder = data_root / "intermediaries" / "raw_jsons" / index
    output_folder = data_root / "intermediaries" / "structured_calls" / index

    logger.info(f"Index: {index}")
    logger.info(f"DATA_ROOT: {data_root}")
    logger.info(f"PDF root: {pdf_root}")
    logger.info(f"Input (raw_jsons): {input_folder}")
    logger.info(f"Output (structured_calls): {output_folder}")

    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF root folder does not exist: {pdf_root}")
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Build PDF metadata lookup (ticker, date, company, quarter, etc.)
    # -------------------------------------------------------------------------
    logger.info(f"Building metadata lookup from PDFs in {pdf_root}")
    metadata_lookup = build_metadata_lookup(pdf_root, index=index)

    logger.info(
        f"Found metadata for {len(metadata_lookup)} PDFs. Structuring transcripts "
        f"from {input_folder} → {output_folder}"
    )

    # -------------------------------------------------------------------------
    # Process and structure the transcripts
    # -------------------------------------------------------------------------
    structure_all_transcripts_from_parts(
        input_folder=input_folder,
        output_folder=output_folder,
        metadata_lookup=metadata_lookup,
    )

    logger.info(f"Finished building structured transcripts for index {index}.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw transcript parts into structured call JSONs."
    )

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Which market index to process.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    structure_transcripts(args.index)


if __name__ == "__main__":
    main()
