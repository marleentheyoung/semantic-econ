#!/usr/bin/env python3
"""
Run full topic→retrieval→indicator→panel pipeline.

Usage:
    python scripts/run_topic_pipeline.py --topic flood --index-dir data/indexes --metadata data/call_metadata.json
"""

import argparse
from pathlib import Path
import pandas as pd

from semantic_measurement.pipeline.topic_runner import TopicRunner
from semantic_measurement.pipeline.call_metadata_loader import load_call_metadata
from semantic_measurement.config.global_calibration import DEFAULT_EMBEDDING_MODEL, DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run topic attention panel pipeline")

    parser.add_argument("--topic", required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/all_calls.json"))
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--embeddings-dir", type=Path)

    parser.add_argument("--queries-root", type=Path, default=Path("data/queries"))
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/panels"),
        help="Directory to write the output panel parquet file",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use memory-efficient streaming mode (recommended for batch processing)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for streaming mode (default: 1000)"
    )

    args = parser.parse_args()

    # Load metadata
    call_metadata = load_call_metadata(args.metadata)

    runner = TopicRunner(
        call_metadata=call_metadata,
        queries_root=args.queries_root,
        index_dir=args.index_dir,
        embeddings_dir=args.embeddings_dir,
        model_name=args.model,
    )

    if args.streaming:
        df = runner.run_topic_streaming(args.topic, batch_size=args.batch_size)
    else:
        df = runner.run_topic(args.topic)

    # Save output
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.topic}_panel.parquet"
    df.to_parquet(out_path)

    print(f"✓ Finished. Saved panel to: {out_path}")
    print(f"Rows: {len(df):,}, Firms: {df['ticker'].nunique():,}")


if __name__ == "__main__":
    main()
