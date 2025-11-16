#!/usr/bin/env python3
"""
Ingestion pipeline: RAW PDFs → structured transcripts → paragraphs →
embeddings → FAISS semantic index → call-level metadata.

This script orchestrates ingestion stages but does NOT contain logic itself.
Each stage is restartable and checks for existing output.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------
# Import pipeline modules (all logic lives in src/)
# ---------------------------------------------------------
from semantic_measurement.pipeline.extract_transcripts import extract_transcripts
from semantic_measurement.pipeline.structure_transcripts import structure_transcripts
from semantic_measurement.pipeline.extract_paragraphs import extract_paragraphs
from semantic_measurement.pipeline.embed_segments import embed_segments
from semantic_measurement.pipeline.build_index import build_faiss_index
from semantic_measurement.pipeline.call_metadata_builder import build_call_metadata_for_index


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def header(msg: str):
    print("\n" + "="*80)
    print(msg)
    print("="*80 + "\n")


# ---------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Measurement Ingestion Pipeline"
    )

    parser.add_argument(
        "--index",
        required=True,
        choices=["SP500", "STOXX600"],
        help="Which market index to process",
    )

    # Flags for each stage
    parser.add_argument("--extract-transcripts", action="store_true",
                        help="Extract raw transcripts from PDFs")
    parser.add_argument("--structure", action="store_true",
                        help="Structure transcripts into JSON format")
    parser.add_argument("--paragraphs", action="store_true",
                        help="Extract paragraphs from structured transcripts")
    parser.add_argument("--embeddings", action="store_true",
                        help="Compute MPNet embeddings and store in chunks")
    parser.add_argument("--faiss", action="store_true",
                        help="Build FAISS semantic index from embeddings")
    parser.add_argument("--metadata", action="store_true",
                        help="Build call-level metadata from snippets")

    parser.add_argument("--all", action="store_true",
                        help="Run ALL ingestion stages")

    args = parser.parse_args()

    index = args.index

    # -----------------------------------------------------
    # Resolve "all" flag into individual stage flags
    # -----------------------------------------------------
    if args.all:
        args.extract_transcripts = True
        args.structure = True
        args.paragraphs = True
        args.embeddings = True
        args.faiss = True
        args.metadata = True

    # -----------------------------------------------------
    # Execute requested stages
    # -----------------------------------------------------

    if args.extract_transcripts:
        header(f"STEP 1: Extract transcripts for {index}")
        extract_transcripts(index)

    if args.structure:
        header(f"STEP 2: Structure transcripts for {index}")
        structure_transcripts(index)

    if args.paragraphs:
        header(f"STEP 3: Extract paragraphs for {index}")
        extract_paragraphs(index)

    if args.embeddings:
        header(f"STEP 4: Compute embeddings for {index}")
        embed_segments(index)

    if args.faiss:
        header(f"STEP 5: Build FAISS index for {index}")
        build_faiss_index(index)

    if args.metadata:
        header(f"STEP 6: Build call-level metadata for {index}")
        build_call_metadata_for_index(index)

    header("Pipeline complete!")
    return 0


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
