from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend

# Should print: "✓ Embedder initialized on device: mps"
embedder = SentenceTransformerBackend()

# Encode some test data
import time
start = time.time()
embeddings = embedder.encode(["test sentence"] * 128, batch_size=64)
elapsed = time.time() - start

print(f"Encoded 128 sentences in {elapsed:.2f}s")
print(f"Embeddings shape: {embeddings.shape}")
