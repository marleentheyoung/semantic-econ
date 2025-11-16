from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.concepts.concept_retriever import ConceptRetriever
from semantic_measurement.config import DATA_ROOT
from pathlib import Path

# 1. Load embedder
embedder = SentenceTransformerBackend(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 2. Create SemanticRetriever (r)
r = SemanticRetriever(
    embedder=embedder,
    data_root=DATA_ROOT,
    index_names=["SP500", "STOXX600"]  # or just ["SP500"]
)

# 3. Create ConceptRetriever using r
cr = ConceptRetriever(
    retriever=r,
    patterns=["flood", "storm surge", "extreme weather"],
    threshold=0.42,
    top_k=200   # how many paragraphs per query to inspect
)

# 4. Get concept-level hits grouped by call_id
hits = cr.retrieve_hits()

print("Number of calls with hits:", len(hits))
print("Example call_ids:", list(hits.keys())[:5])
