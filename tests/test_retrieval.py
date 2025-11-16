# tests/test_retrieval.py

from pathlib import Path

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.config import DATA_ROOT


# tests/test_retrieval.py

from pathlib import Path

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.config import DATA_ROOT

def test_basic_retrieval():
    print("\n=== RUNNING BASIC RETRIEVAL TEST ===")

    # 1. Load embedder
    embedder = SentenceTransformerBackend(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 2. Load retriever
    retriever = SemanticRetriever(
        embedder=embedder,
        data_root=DATA_ROOT,
        index_names=["SP500"],     # keep it simple for testing
    )

    # 3. Search
    query = "climate risk"
    print(f"Searching for: {query!r}\n")

    results = retriever.search_by_text(query, top_k=5)

    # 4. Basic correctness checks
    assert isinstance(results, list), "Expected a list of results"
    assert len(results) > 0, "No retrieval results returned"

    print(f"Returned {len(results)} results:\n")

    # 5. Print results
    for i, r in enumerate(results, start=1):
        score = r["score"]
        idx = r["index_name"]
        text = r["snippet"]["text"].replace("\n", " ")
        preview = text[:520] + ("..." if len(text) > 520 else "")

        print(f"Result {i}:")
        print(f"  Score:      {score:.4f}")
        print(f"  Index:      {idx}")
        print(f"  Paragraph:  {preview}")
        print()

    print("=== TEST COMPLETED SUCCESSFULLY ===\n")

test_basic_retrieval()