# src/semantic_measurement/retrieval/semantic_retriever.py

from __future__ import annotations
import faiss
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


class SemanticRetriever:

    def __init__(
        self,
        embedder,
        index_dir: Path,
        embeddings_dir: Path,
        index_names: List[str],
    ):
        self.embedder = embedder
        self.indexes = {}

        for name in index_names:
            faiss_path = index_dir / name / "semantic_index.faiss"
            snippets_path = embeddings_dir / f"{name}_mpnet" / "snippets.parquet"  # CHANGED: .json → .parquet

            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS index missing: {faiss_path}")

            if not snippets_path.exists():
                raise FileNotFoundError(f"snippets.parquet missing: {snippets_path}")

            index = faiss.read_index(str(faiss_path))

            # CHANGED: Read from Parquet and convert to list of dicts
            table = pq.read_table(snippets_path)
            snippets = table.to_pylist()

            self.indexes[name] = {
                "faiss_index": index,
                "snippets": snippets,
            }

    # ---------------------------------------------------------
    # Core embedding-based search
    # ---------------------------------------------------------
    def search_by_embedding(self, query_vec: np.ndarray, top_k: int = 100):
        if query_vec.ndim == 1:
            q = query_vec.reshape(1, -1)
        else:
            q = query_vec

        all_results = []

        for name, bundle in self.indexes.items():
            index = bundle["faiss_index"]
            snippets = bundle["snippets"]

            scores, ids = index.search(q, top_k)
            scores, ids = scores[0], ids[0]

            for score, fid in zip(scores, ids):
                if fid < 0:
                    continue

                all_results.append({
                    "score": float(score),
                    "faiss_id": int(fid),
                    "index_name": name,
                    "snippet": snippets[fid],
                })

        # sort global
        all_results.sort(key=lambda r: r["score"], reverse=True)
        return all_results[:top_k]

    # ---------------------------------------------------------
    # Text → Embedding → Search
    # ---------------------------------------------------------
    def search_by_text(self, text: str, top_k: int = 100, batch_size: int = 32):
        emb = self.embedder.encode([text], batch_size=batch_size)[0]
        return self.search_by_embedding(emb, top_k=top_k)