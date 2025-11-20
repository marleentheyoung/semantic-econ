# tests/test_backward_compatibility.py

"""
Backward compatibility tests.

Ensures that existing code continues to work after adding streaming APIs.
All tests should pass without modifications to calling code.
"""

import pytest
from pathlib import Path

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.concepts.concept_retriever import ConceptRetriever
from semantic_measurement.indicators.indicator_builder import IndicatorBuilder, ParagraphHit
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.config import DATA_ROOT


# ============================================================================
# Test 1: Original test_multi_query.py still works
# ============================================================================

def test_original_multi_query_pattern():
    """
    Reproduce the exact pattern from tests/test_multi_query.py.
    This should work without any changes.
    """
    
    # 1. Load embedder
    embedder = SentenceTransformerBackend(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 2. Create SemanticRetriever (r)
    r = SemanticRetriever(
        embedder=embedder,
        data_root=DATA_ROOT,
        index_names=["SP500", "STOXX600"]
    )

    # 3. Create ConceptRetriever using r
    cr = ConceptRetriever(
        retriever=r,
        patterns=["flood", "storm surge", "extreme weather"],
        threshold=0.42,
        top_k=200
    )

    # 4. Get concept-level hits grouped by call_id
    hits = cr.retrieve_hits()

    # Assertions
    assert isinstance(hits, dict), "Should return dict"
    assert all(isinstance(v, list) for v in hits.values()), "Values should be lists"
    
    if hits:
        first_call_hits = list(hits.values())[0]
        assert all(isinstance(h, ParagraphHit) for h in first_call_hits), \
            "List items should be ParagraphHit instances"
    
    print(f"✓ Original test_multi_query.py pattern works")
    print(f"  Found {len(hits)} calls with hits")


# ============================================================================
# Test 2: Original test_indicator.py still works
# ============================================================================

def test_original_indicator_pattern():
    """
    Reproduce the exact pattern from tests/test_indicator.py.
    This should work without any changes.
    """
    
    # Fake one-call metadata (from original test)
    meta = {
        "CALL123": {
            "total_snippets": 100,
            "total_sentences": 300,
            "management_sentences": 180,
            "qa_sentences": 120,
            "management_snippets": 60,
            "qa_snippets": 40,
        }
    }

    # Fake hits (from original test)
    hits = {
        "CALL123": [
            ParagraphHit(faiss_id=1, similarity=0.60, sentence_count=4, section="management", call_id="CALL123"),
            ParagraphHit(faiss_id=2, similarity=0.55, sentence_count=3, section="qa", call_id="CALL123"),
        ]
    }

    ib = IndicatorBuilder(meta)
    out = ib.build_indicators(hits)

    # Assertions
    assert "CALL123" in out, "Should have indicators for CALL123"
    
    indicators = out["CALL123"]
    
    # Check expected fields exist
    expected_fields = [
        "exposure", "avgSim", "intensity",
        "mgmt_exposure", "mgmt_avgSim", "mgmt_intensity",
        "qa_exposure", "qa_avgSim", "qa_intensity",
        "n_hits_total", "n_hits_mgmt", "n_hits_qa",
    ]
    
    for field in expected_fields:
        assert field in indicators, f"Missing field: {field}"
    
    # Check values are reasonable
    assert 0 <= indicators["exposure"] <= 1, "Exposure should be in [0,1]"
    assert 0 <= indicators["avgSim"] <= 1, "avgSim should be in [0,1]"
    assert indicators["n_hits_total"] == 2, "Should have 2 total hits"
    assert indicators["n_hits_mgmt"] == 1, "Should have 1 mgmt hit"
    assert indicators["n_hits_qa"] == 1, "Should have 1 qa hit"
    
    print(f"✓ Original test_indicator.py pattern works")


# ============================================================================
# Test 3: Existing TopicRunner.run_topic() still works
# ============================================================================

def test_topic_runner_original_api():
    """
    Verify TopicRunner.run_topic() (original API) still works.
    """
    import json
    from semantic_measurement.pipeline.topic_runner import TopicRunner
    
    # Check if we have the required files
    metadata_file = DATA_ROOT / "metadata" / "SP500_calls.json"
    
    if not metadata_file.exists():
        pytest.skip("Metadata file not found, skipping test")
    
    with metadata_file.open("r") as f:
        call_metadata = json.load(f)
    
    runner = TopicRunner(
        call_metadata=call_metadata,
        queries_root=DATA_ROOT / "queries",
        index_dir=DATA_ROOT / "indexes",
        embeddings_dir=DATA_ROOT / "embeddings",
    )
    
    # Check if flood topic exists
    flood_query = DATA_ROOT / "queries" / "flood.json"
    if not flood_query.exists():
        pytest.skip("flood.json not found, skipping test")
    
    # Run original API
    df = runner.run_topic("flood")
    
    # Assertions
    assert df is not None, "Should return DataFrame"
    assert len(df) > 0, "Should have rows"
    assert "topic" in df.columns, "Should have topic column"
    assert "exposure" in df.columns, "Should have exposure column"
    
    print(f"✓ TopicRunner.run_topic() works")
    print(f"  Returned {len(df)} rows")


# ============================================================================
# Test 4: ConceptRetriever signature hasn't changed
# ============================================================================

def test_concept_retriever_signature():
    """Verify ConceptRetriever constructor signature is unchanged."""
    
    import inspect
    
    sig = inspect.signature(ConceptRetriever.__init__)
    params = list(sig.parameters.keys())
    
    # Expected parameters (excluding 'self')
    expected = ['self', 'retriever', 'patterns', 'threshold', 'top_k']
    
    assert params == expected, \
        f"ConceptRetriever signature changed: {params} != {expected}"
    
    print(f"✓ ConceptRetriever signature unchanged")


# ============================================================================
# Test 5: IndicatorBuilder signature hasn't changed
# ============================================================================

def test_indicator_builder_signature():
    """Verify IndicatorBuilder constructor signature is unchanged."""
    
    import inspect
    
    sig = inspect.signature(IndicatorBuilder.__init__)
    params = list(sig.parameters.keys())
    
    expected = ['self', 'call_metadata']
    
    assert params == expected, \
        f"IndicatorBuilder signature changed: {params} != {expected}"
    
    print(f"✓ IndicatorBuilder signature unchanged")


# ============================================================================
# Test 6: Return types haven't changed
# ============================================================================

def test_return_types_unchanged():
    """Verify that original methods return the same types."""
    
    # Minimal setup
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
        patterns=["test"],
        threshold=0.42,
        top_k=10,
    )
    
    # Test retrieve_hits return type
    hits = concept.retrieve_hits()
    assert isinstance(hits, dict), "retrieve_hits() should return dict"
    
    # Test IndicatorBuilder return type
    meta = {
        "TEST": {
            "total_sentences": 100,
            "management_sentences": 60,
            "qa_sentences": 40,
        }
    }
    
    builder = IndicatorBuilder(meta)
    indicators = builder.build_indicators({})
    assert isinstance(indicators, dict), "build_indicators() should return dict"
    
    print(f"✓ Return types unchanged")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BACKWARD COMPATIBILITY TEST SUITE")
    print("="*70 + "\n")
    
    pytest.main([__file__, "-v", "-s"])