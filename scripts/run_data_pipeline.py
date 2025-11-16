#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from semantic_measurement.config import DATA_ROOT
from semantic_measurement.data.pdf_preprocessing import (
    extract_transcripts_memory_efficient,
    get_optimal_batch_sizes,
)
from semantic_measurement.data.pdf_metadata import build_metadata_lookup
from semantic_measurement.data.structure_transcripts import (
    structure_all_transcripts_from_parts,
)
from semantic_measurement.preprocessing.paragraph_extractor import (
    extract_paragraphs_from_folder,
)


def run_pipeline(
    index: str,
    num_parts: int | None = None,
    memory_limit_gb: float = 4.0,
) -> None:
    logger = logging.getLogger(__name__)

    # -----------------------------
    # 0. Derive all paths from config
    # -----------------------------
    data_root: Path = DATA_ROOT

    pdf_root = data_root / "dataset" / "raw" / index
    transcripts_parts_root = (
        data_root / "dataset" / "intermediaries" / "transcripts_parts" / index
    )
    structured_calls_root = (
        data_root / "dataset" / "intermediaries" / "structured_calls" / index
    )
    paragraphs_root = data_root / "dataset" / "intermediaries" / "paragraphs"
    paragraphs_root.mkdir(parents=True, exist_ok=True)
    paragraphs_file = paragraphs_root / f"{index}_paragraphs.jsonl"

    logger.info("Index: %s", index)
    logger.info("DATA_ROOT: %s", data_root)
    logger.info("PDF root: %s", pdf_root)
    logger.info("Transcripts parts: %s", transcripts_parts_root)
    logger.info("Structured calls: %s", structured_calls_root)
    logger.info("Paragraphs file: %s", paragraphs_file)

    if not pdf_root.exists():
        logger.error("PDF root folder does not exist: %s", pdf_root)
        return

    transcripts_parts_root.mkdir(parents=True, exist_ok=True)
    structured_calls_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. PDFs → transcripts_parts
    # -----------------------------
    logger.info("Step 1: Extracting transcripts from PDFs (memory-efficient).")

    batch_cfg = get_optimal_batch_sizes()
    logger.info("Automatic batch size config: %s", batch_cfg)

    # If num_parts not explicitly given, choose something reasonable
    if num_parts is None:
        # You can tweak this heuristic if you like
        base_parts = 10
        multiplier = batch_cfg.get("num_parts_multiplier", 1.0)
        num_parts = max(5, int(base_parts * multiplier))

    logger.info("Using num_parts = %d", num_parts)

    output_basename = transcripts_parts_root / "transcripts_data"

    extract_transcripts_memory_efficient(
        pdf_root_folder=pdf_root,
        output_basename=output_basename,
        num_parts=num_parts,
        memory_limit_gb=memory_limit_gb,
    )

    logger.info("✅ Finished extracting transcripts.")

    # -----------------------------
    # 2. transcripts_parts → structured_calls
    # -----------------------------
    logger.info("Step 2: Building metadata lookup from PDFs.")
    metadata_lookup = build_metadata_lookup(pdf_root, index=index)
    logger.info("Metadata entries: %d", len(metadata_lookup))

    logger.info("Step 2: Structuring transcripts.")
    structure_all_transcripts_from_parts(
        input_folder=transcripts_parts_root,
        output_folder=structured_calls_root,
        metadata_lookup=metadata_lookup,
    )
    logger.info("✅ Finished structuring transcripts.")

    # -----------------------------
    # 3. structured_calls → paragraphs JSONL
    # -----------------------------
    logger.info("Step 3: Extracting paragraph-level records.")
    extract_paragraphs_from_folder(
        input_folder=str(structured_calls_root),
        output_file=str(paragraphs_file),
    )
    logger.info("✅ Finished extracting paragraphs to %s", paragraphs_file)

    logger.info("🎉 Full data pipeline completed for index %s", index)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full data pipeline: PDFs → transcripts_parts → "
            "structured_calls → paragraphs JSONL (index-aware, config-driven)."
        )
    )

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Market index to process (SP500 or STOXX600).",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=None,
        help="Optional override for number of transcript parts. "
             "If omitted, a heuristic based on available memory is used.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=4.0,
        help="Approximate memory limit in GB for PDF extraction (default: 4.0).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_pipeline(
        index=args.index,
        num_parts=args.num_parts,
        memory_limit_gb=args.memory_limit_gb,
    )


if __name__ == "__main__":
    main()
