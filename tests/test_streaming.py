# tests/test_streaming.py

"""
Test streaming APIs for ConceptRetriever and IndicatorBuilder.

Verifies that streaming methods produce identical results to batch methods.
"""

import pytest
from pathlib import Path

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.concepts.concept_retriever import ConceptRetriever
from semantic_measurement.indicators.indicator_builder import IndicatorBuilder
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.config import DATA_ROOT


@pytest.fixture
def embedder():
    """Shared embedder for all tests."""
    return SentenceTransformerBackend(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


@pytest.fixture
def retriever(embedder):
    """Shared retriever for all tests."""
    return SemanticRetriever(
        embedder=embedder,
        data_root=DATA_ROOT,
        index_names=["SP500"],  # Use smaller index for faster tests
    )


@pytest.fixture
def sample_metadata():
    """Minimal metadata for testing."""
    return {
        "CALL_001": {
            "total_sentences": 100,
            "management_sentences": 60,
            "qa_sentences": 40,
            "ticker": "AAPL",
            "year": 2023,
            "quarter": 1,
        },
        "CALL_002": {
            "total_sentences": 150,
            "management_sentences": 90,
            "qa_sentences": 60,
            "ticker": "MSFT",
            "year": 2023,
            "quarter": 2,
        },
    }


# ============================================================================
# Test 1: ConceptRetriever streaming produces same results as batch
# ============================================================================

def test_concept_retriever_streaming_matches_batch(retriever):
    """Verify streaming and batch APIs return identical hits."""
    
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=["climate risk", "flooding"],
        threshold=0.42,
        top_k=50,
    )
    
    # Get results from batch API
    batch_hits = concept.retrieve_hits()
    
    # Get results from streaming API
    streaming_hits = {}
    for call_id, hits in concept.retrieve_hits_streaming():
        streaming_hits[call_id] = hits
    
    # Verify same call_ids
    assert set(batch_hits.keys()) == set(streaming_hits.keys()), \
        "Streaming and batch APIs should return same call_ids"
    
    # Verify same number of hits per call
    for call_id in batch_hits:
        batch_count = len(batch_hits[call_id])
        stream_count = len(streaming_hits[call_id])
        assert batch_count == stream_count, \
            f"Call {call_id}: batch={batch_count}, streaming={stream_count}"
    
    # Verify same faiss_ids (order may differ, so use sets)
    for call_id in batch_hits:
        batch_ids = {h.faiss_id for h in batch_hits[call_id]}
        stream_ids = {h.faiss_id for h in streaming_hits[call_id]}
        assert batch_ids == stream_ids, \
            f"Call {call_id}: different faiss_ids between batch and streaming"
    
    print(f"✓ Test passed: {len(batch_hits)} calls, identical results")


# ============================================================================
# Test 2: ConceptRetriever batched streaming reassembles correctly
# ============================================================================

def test_concept_retriever_batched_reassembles(retriever):
    """Verify batched streaming can be reassembled into full results."""
    
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=["climate risk"],
        threshold=0.42,
        top_k=100,
    )
    
    # Get original results
    original_hits = concept.retrieve_hits()
    
    # Get batched results and reassemble
    reassembled_hits = {}
    for batch in concept.retrieve_hits_batched(batch_size=5):
        reassembled_hits.update(batch)
    
    # Verify identical
    assert set(original_hits.keys()) == set(reassembled_hits.keys())
    
    for call_id in original_hits:
        orig_ids = {h.faiss_id for h in original_hits[call_id]}
        reasm_ids = {h.faiss_id for h in reassembled_hits[call_id]}
        assert orig_ids == reasm_ids
    
    print(f"✓ Test passed: Batched streaming reassembles correctly")


# ============================================================================
# Test 3: IndicatorBuilder streaming produces same results as batch
# ============================================================================

def test_indicator_builder_streaming_matches_batch(retriever, sample_metadata):
    """Verify IndicatorBuilder streaming produces identical indicators."""
    
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=["climate risk"],
        threshold=0.42,
        top_k=50,
    )
    
    hits = concept.retrieve_hits()
    
    builder = IndicatorBuilder(sample_metadata)
    
    # Batch API
    batch_indicators = builder.build_indicators(hits)
    
    # Streaming API
    streaming_indicators = {}
    for call_id, ind in builder.build_indicators_streaming(
        concept.retrieve_hits_streaming()
    ):
        streaming_indicators[call_id] = ind
    
    # Verify same call_ids
    assert set(batch_indicators.keys()) == set(streaming_indicators.keys())
    
    # Verify same indicator values
    for call_id in batch_indicators:
        batch_ind = batch_indicators[call_id]
        stream_ind = streaming_indicators[call_id]
        
        # Check all numeric fields match (within floating point precision)
        for key in batch_ind:
            if isinstance(batch_ind[key], (int, float)):
                assert abs(batch_ind[key] - stream_ind[key]) < 1e-6, \
                    f"Call {call_id}, field {key}: {batch_ind[key]} != {stream_ind[key]}"
    
    print(f"✓ Test passed: IndicatorBuilder streaming matches batch")


# ============================================================================
# Test 4: Empty results handling
# ============================================================================

def test_streaming_handles_empty_results(retriever, sample_metadata):
    """Verify streaming APIs handle empty results gracefully."""
    
    # Use a pattern unlikely to match anything
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=["xyzabc123nonexistent"],
        threshold=0.99,  # Very high threshold
        top_k=10,
    )
    
    # Batch API
    batch_hits = concept.retrieve_hits()
    
    # Streaming API
    streaming_hits = list(concept.retrieve_hits_streaming())
    
    # Both should be empty or very small
    assert len(batch_hits) == len(streaming_hits), \
        "Batch and streaming should handle empty results identically"
    
    print(f"✓ Test passed: Empty results handled correctly")


# ============================================================================
# Test 5: Large batch size edge case
# ============================================================================

def test_batched_with_large_batch_size(retriever):
    """Verify batched streaming works when batch_size > total results."""
    
    concept = ConceptRetriever(
        retriever=retriever,
        patterns=["climate risk"],
        threshold=0.42,
        top_k=50,
    )
    
    # Get original
    original = concept.retrieve_hits()
    
    # Use batch size larger than total results
    batches = list(concept.retrieve_hits_batched(batch_size=10000))
    
    # Should produce exactly 1 batch
    assert len(batches) == 1, \
        f"Expected 1 batch with large batch_size, got {len(batches)}"
    
    # Batch should equal original
    assert set(batches[0].keys()) == set(original.keys())
    
    print(f"✓ Test passed: Large batch size handled correctly")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STREAMING API CORRECTNESS TESTS")
    print("="*70 + "\n")
    
    pytest.main([__file__, "-v", "-s"])