#!/usr/bin/env python

"""
Transcript extraction pipeline module.

Provides two entry points:
- extract_transcripts(): importable function API
- main(): CLI interface
"""

import argparse
import logging
from pathlib import Path

from semantic_measurement.config.global_calibration import DATA_ROOT
from semantic_measurement.data.pdf_preprocessing import (
    extract_transcripts_memory_efficient,
    get_optimal_batch_sizes,
)


def extract_transcripts(index: str, num_parts: int = 10, memory_limit_gb: float = 4.0):
    """
    Importable pipeline function for extracting transcripts.

    Parameters
    ----------
    index : str
        "SP500" or "STOXX600"
    num_parts : int
    memory_limit_gb : float
    """
    logger = logging.getLogger(__name__)

    data_root: Path = DATA_ROOT

    # pdf_root = data_root / "raw" / index
    pdf_root = Path("/Users/marleendejonge/Desktop/ECC-data-generation/data/raw/") / index
    output_folder = data_root / "intermediaries" / "raw_jsons" / index
    output_folder.mkdir(parents=True, exist_ok=True)
    output_basename = output_folder / "transcripts_data"

    logger.info(f"Index: {index}")
    logger.info(f"DATA_ROOT: {data_root}")
    logger.info(f"PDF root: {pdf_root}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Output basename: {output_basename}")

    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF root folder does not exist: {pdf_root}")

    batch_cfg = get_optimal_batch_sizes()
    logger.info(f"Automatic batch size configuration: {batch_cfg}")

    extract_transcripts_memory_efficient(
        pdf_root_folder=str(pdf_root),
        output_basename=str(output_basename),
        num_parts=num_parts,
        memory_limit_gb=memory_limit_gb,
    )

    logger.info(f"Finished extracting transcripts for index {index}.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcripts from raw PDF earnings calls "
                    "(memory-efficient, index-aware)."
    )

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Market index to process.",
    )
    parser.add_argument("--num-parts", type=int, default=10)
    parser.add_argument("--memory-limit-gb", type=float, default=4.0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    extract_transcripts(
        index=args.index,
        num_parts=args.num_parts,
        memory_limit_gb=args.memory_limit_gb
    )


if __name__ == "__main__":
    main()
