# tests/test_memory_efficiency.py

"""
Memory efficiency benchmarks for streaming vs. batch processing.

These tests measure actual memory usage to verify streaming reduces memory footprint.
Run with: python tests/test_memory_efficiency.py
"""

import tracemalloc
import time
from pathlib import Path

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.concepts.concept_retriever import ConceptRetriever
from semantic_measurement.indicators.indicator_builder import IndicatorBuilder
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.pipeline.topic_runner import TopicRunner
from semantic_measurement.config import DATA_ROOT


def format_bytes(bytes_value):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"


def measure_memory(func, *args, **kwargs):
    """
    Measure peak memory usage of a function.
    
    Returns:
        (result, current_memory, peak_memory)
    """
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, current, peak


# ============================================================================
# Benchmark 1: ConceptRetriever batch vs. streaming
# ============================================================================

def benchmark_concept_retriever():
    """Compare memory usage: batch vs. streaming retrieval."""
    
    print("\n" + "="*70)
    print("BENCHMARK 1: ConceptRetriever Memory Usage")
    print("="*70)
    
    embedder = SentenceTransformerBackend(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    retriever = SemanticRetriever(
        embedder=embedder,
        data_root=DATA_ROOT,
        index_names=["SP500"],
    )
    
    # Test with multiple patterns (more realistic)
    patterns = [
        "climate risk",
        "flooding",
        "extreme weather",
        "sea level rise",
        "carbon emissions",
    ]
    
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=patterns,
        threshold=0.42,
        top_k=200,
    )
    
    # -------------------------------------------------------------------------
    # Batch API
    # -------------------------------------------------------------------------
    print("\n[1/2] Testing batch API...")
    start_time = time.time()
    
    batch_hits, batch_current, batch_peak = measure_memory(
        concept.retrieve_hits
    )
    
    batch_time = time.time() - start_time
    
    print(f"  ✓ Completed in {batch_time:.2f}s")
    print(f"  Current memory: {format_bytes(batch_current)}")
    print(f"  Peak memory:    {format_bytes(batch_peak)}")
    print(f"  Calls found:    {len(batch_hits)}")
    
    # -------------------------------------------------------------------------
    # Streaming API
    # -------------------------------------------------------------------------
    print("\n[2/2] Testing streaming API...")
    start_time = time.time()
    
    def consume_stream():
        """Consume streaming results (simulating real usage)."""
        hits = {}
        for call_id, call_hits in concept.retrieve_hits_streaming():
            hits[call_id] = call_hits
        return hits
    
    stream_hits, stream_current, stream_peak = measure_memory(consume_stream)
    
    stream_time = time.time() - start_time
    
    print(f"  ✓ Completed in {stream_time:.2f}s")
    print(f"  Current memory: {format_bytes(stream_current)}")
    print(f"  Peak memory:    {format_bytes(stream_peak)}")
    print(f"  Calls found:    {len(stream_hits)}")
    
    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    memory_reduction = (1 - stream_peak / batch_peak) * 100
    time_difference = ((stream_time - batch_time) / batch_time) * 100
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Peak memory reduction:  {memory_reduction:+.1f}%")
    print(f"  Time difference:        {time_difference:+.1f}%")
    
    if memory_reduction > 0:
        print(f"  ✓ Streaming uses {memory_reduction:.1f}% less peak memory")
    else:
        print(f"  ⚠ Streaming uses {abs(memory_reduction):.1f}% more peak memory")
    
    return {
        "batch_peak": batch_peak,
        "stream_peak": stream_peak,
        "reduction_pct": memory_reduction,
    }


# ============================================================================
# Benchmark 2: Full pipeline batch vs. streaming
# ============================================================================

def benchmark_full_pipeline():
    """Compare memory usage: TopicRunner batch vs. streaming."""
    
    print("\n" + "="*70)
    print("BENCHMARK 2: Full Pipeline Memory Usage")
    print("="*70)
    
    # Load minimal metadata for testing
    import json
    metadata_file = DATA_ROOT / "metadata" / "SP500_calls.json"
    
    if not metadata_file.exists():
        print("  ⚠ Metadata file not found, skipping full pipeline benchmark")
        return None
    
    with metadata_file.open("r") as f:
        call_metadata = json.load(f)
    
    runner = TopicRunner(
        call_metadata=call_metadata,
        queries_root=DATA_ROOT / "queries",
        index_dir=DATA_ROOT / "indexes",
        embeddings_dir=DATA_ROOT / "embeddings",
        model_name="sentence-transformers/all-mpnet-base-v2",
    )
    
    # -------------------------------------------------------------------------
    # Check if flood topic exists
    # -------------------------------------------------------------------------
    flood_query = DATA_ROOT / "queries" / "flood.json"
    if not flood_query.exists():
        print("  ⚠ flood.json not found, skipping full pipeline benchmark")
        return None
    
    # -------------------------------------------------------------------------
    # Batch version
    # -------------------------------------------------------------------------
    print("\n[1/2] Testing batch pipeline...")
    start_time = time.time()
    
    batch_df, batch_current, batch_peak = measure_memory(
        runner.run_topic,
        "flood"
    )
    
    batch_time = time.time() - start_time
    
    print(f"  ✓ Completed in {batch_time:.2f}s")
    print(f"  Current memory: {format_bytes(batch_current)}")
    print(f"  Peak memory:    {format_bytes(batch_peak)}")
    print(f"  Panel rows:     {len(batch_df)}")
    
    # -------------------------------------------------------------------------
    # Streaming version
    # -------------------------------------------------------------------------
    print("\n[2/2] Testing streaming pipeline...")
    start_time = time.time()
    
    stream_df, stream_current, stream_peak = measure_memory(
        runner.run_topic_streaming,
        "flood",
        batch_size=500
    )
    
    stream_time = time.time() - start_time
    
    print(f"  ✓ Completed in {stream_time:.2f}s")
    print(f"  Current memory: {format_bytes(stream_current)}")
    print(f"  Peak memory:    {format_bytes(stream_peak)}")
    print(f"  Panel rows:     {len(stream_df)}")
    
    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    memory_reduction = (1 - stream_peak / batch_peak) * 100
    time_difference = ((stream_time - batch_time) / batch_time) * 100
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Peak memory reduction:  {memory_reduction:+.1f}%")
    print(f"  Time difference:        {time_difference:+.1f}%")
    
    if memory_reduction > 0:
        print(f"  ✓ Streaming pipeline uses {memory_reduction:.1f}% less peak memory")
    else:
        print(f"  ⚠ Streaming pipeline uses {abs(memory_reduction):.1f}% more peak memory")
    
    return {
        "batch_peak": batch_peak,
        "stream_peak": stream_peak,
        "reduction_pct": memory_reduction,
    }


# ============================================================================
# Benchmark 3: Batched streaming with different batch sizes
# ============================================================================

def benchmark_batch_sizes():
    """Test how batch size affects memory usage."""
    
    print("\n" + "="*70)
    print("BENCHMARK 3: Batch Size Impact")
    print("="*70)
    
    embedder = SentenceTransformerBackend(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    retriever = SemanticRetriever(
        embedder=embedder,
        data_root=DATA_ROOT,
        index_names=["SP500"],
    )
    
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=["climate risk", "flooding"],
        threshold=0.42,
        top_k=200,
    )
    
    batch_sizes = [100, 500, 1000, 5000]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n[Testing batch_size={batch_size}]")
        
        def consume_batched():
            hits = {}
            for batch in concept.retrieve_hits_batched(batch_size=batch_size):
                hits.update(batch)
            return hits
        
        _, current, peak = measure_memory(consume_batched)
        
        print(f"  Peak memory: {format_bytes(peak)}")
        
        results.append({
            "batch_size": batch_size,
            "peak_memory": peak,
        })
    
    print("\n" + "-"*70)
    print("SUMMARY:")
    print(f"  {'Batch Size':<15} {'Peak Memory':<20} {'vs. Smallest':<15}")
    print("-"*70)
    
    min_peak = min(r["peak_memory"] for r in results)
    
    for r in results:
        increase = ((r["peak_memory"] - min_peak) / min_peak) * 100
        print(f"  {r['batch_size']:<15} {format_bytes(r['peak_memory']):<20} {increase:+.1f}%")
    
    return results


# ============================================================================
# Main runner
# ============================================================================

def run_all_benchmarks():
    """Run all memory efficiency benchmarks."""
    
    print("\n" + "="*70)
    print("MEMORY EFFICIENCY BENCHMARK SUITE")
    print("="*70)
    print("\nThis will measure memory usage for batch vs. streaming APIs.")
    print("Each test may take 1-2 minutes to complete.\n")
    
    results = {}
    
    # Benchmark 1
    try:
        results["concept_retriever"] = benchmark_concept_retriever()
    except Exception as e:
        print(f"\n⚠ Benchmark 1 failed: {e}")
    
    # Benchmark 2
    try:
        results["full_pipeline"] = benchmark_full_pipeline()
    except Exception as e:
        print(f"\n⚠ Benchmark 2 failed: {e}")
    
    # Benchmark 3
    try:
        results["batch_sizes"] = benchmark_batch_sizes()
    except Exception as e:
        print(f"\n⚠ Benchmark 3 failed: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if "concept_retriever" in results and results["concept_retriever"]:
        print(f"\nConceptRetriever:")
        print(f"  Memory reduction: {results['concept_retriever']['reduction_pct']:.1f}%")
    
    if "full_pipeline" in results and results["full_pipeline"]:
        print(f"\nFull Pipeline:")
        print(f"  Memory reduction: {results['full_pipeline']['reduction_pct']:.1f}%")
    
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()