#!/usr/bin/env python3
"""
Run multiple topics for time series validation analysis.

This script processes multiple topics and saves panel datasets
for subsequent time series analysis.

Usage:
    python scripts/run_validation_topics.py --output outputs/validation
    python scripts/run_validation_topics.py --streaming --output outputs/validation
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import logging

from semantic_measurement.pipeline.topic_runner import TopicRunner
from semantic_measurement.pipeline.call_metadata_loader import load_call_metadata
from semantic_measurement.config.global_calibration import (
    DEFAULT_EMBEDDING_MODEL, 
    DATA_ROOT
)


# ============================================================================
# Configuration
# ============================================================================

VALIDATION_TOPICS = [
    "climate_energy_transition",
    "artificial_intelligence",
    "geopolitical_risk",
]


# ============================================================================
# Main Processing Logic
# ============================================================================

def process_topics(
    topics: list,
    output_dir: Path,
    metadata_path: Path,
    streaming: bool = False,
    batch_size: int = 1000,
):
    """
    Process multiple topics and save results.
    
    Parameters
    ----------
    topics : list
        List of topic names to process
    output_dir : Path
        Directory to save output files
    metadata_path : Path
        Path to call metadata JSON
    streaming : bool
        Use streaming API for memory efficiency
    batch_size : int
        Batch size for streaming mode
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata once
    logger.info(f"Loading call metadata from {metadata_path}")
    call_metadata = load_call_metadata(metadata_path)
    logger.info(f"Loaded metadata for {len(call_metadata)} calls")
    
    # Initialize runner
    runner = TopicRunner(
        call_metadata=call_metadata,
        queries_root=DATA_ROOT / "queries",
        index_dir=DATA_ROOT / "indexes",
        embeddings_dir=DATA_ROOT / "embeddings",
        model_name=DEFAULT_EMBEDDING_MODEL,
    )
    
    # Process each topic
    results = {}
    
    for i, topic in enumerate(topics, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing topic {i}/{len(topics)}: {topic}")
        logger.info(f"{'='*70}\n")
        
        try:
            # Run pipeline
            if streaming:
                logger.info(f"Using streaming mode (batch_size={batch_size})")
                df = runner.run_topic_streaming(topic, batch_size=batch_size)
            else:
                logger.info("Using batch mode")
                df = runner.run_topic(topic)
            
            # Save individual topic file
            output_file = output_dir / f"{topic}_panel.parquet"
            df.to_parquet(output_file, index=False)
            
            logger.info(f"✓ Saved {topic} panel: {len(df)} rows")
            logger.info(f"  File: {output_file}")
            
            # Store for combined output
            results[topic] = df
            
        except Exception as e:
            logger.error(f"✗ Failed to process {topic}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # -------------------------------------------------------------------------
    # Create combined dataset with all topics
    # -------------------------------------------------------------------------
    if results:
        logger.info(f"\n{'='*70}")
        logger.info("Creating combined dataset")
        logger.info(f"{'='*70}\n")
        
        combined = pd.concat(results.values(), ignore_index=True)
        combined_file = output_dir / "validation_topics_combined.parquet"
        combined.to_parquet(combined_file, index=False)
        
        logger.info(f"✓ Saved combined dataset: {len(combined)} rows")
        logger.info(f"  File: {combined_file}")
        
        # Summary statistics
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        
        for topic in topics:
            if topic in results:
                df = results[topic]
                logger.info(f"\n{topic}:")
                logger.info(f"  Rows: {len(df):,}")
                logger.info(f"  Firms: {df['ticker'].nunique()}")
                logger.info(f"  Year range: {df['year'].min()}-{df['year'].max()}")
                logger.info(f"  Mean exposure: {df['exposure'].mean():.4f}")
                logger.info(f"  Calls with hits: {(df['n_hits_total'] > 0).sum():,}")
    
    return results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process validation topics for time series analysis"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation"),
        help="Output directory for panel datasets"
    )
    
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DATA_ROOT / "metadata" / "all_calls.json",
        help="Path to call metadata file"
    )
    
    parser.add_argument(
        "--topics",
        nargs="+",
        default=VALIDATION_TOPICS,
        help="Topics to process (default: climate, AI, geopolitical)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for memory efficiency"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for streaming mode"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check metadata file exists
    if not args.metadata.exists():
        logging.error(f"Metadata file not found: {args.metadata}")
        logging.error("Run scripts/merge_call_metadata.py first")
        return 1
    
    # Process topics
    process_topics(
        topics=args.topics,
        output_dir=args.output,
        metadata_path=args.metadata,
        streaming=args.streaming,
        batch_size=args.batch_size,
    )
    
    logging.info("\n✓ All topics processed successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())