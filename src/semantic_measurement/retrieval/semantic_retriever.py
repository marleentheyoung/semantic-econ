# src/semantic_measurement/retrieval/semantic_retriever.py

from __future__ import annotations
import faiss
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


class SemanticRetriever:
    """
    Unified semantic retriever for multiple FAISS indices
    (e.g., SP500 + STOXX600).

    Loads:
      - FAISS index from data/indexes/{index}/semantic_index.faiss
      - snippets from   data/embeddings/{index}_mpnet/snippets.json
    """

    def __init__(
        self,
        embedder,
        data_root: Path,
        index_names: List[str],
    ):
        """
        Parameters
        ----------
        embedder : SentenceTransformerBackend
        data_root : Path
            The project's DATA_ROOT directory
        index_names : list[str]
        """
        self.embedder = embedder
        self.indexes = {}

        for name in index_names:

            # FAISS FILE
            faiss_path = (
                data_root / "indexes" / name / "semantic_index.faiss"
            )

            # SNIPPETS FILE
            snippets_path = (
                data_root / "embeddings" / f"{name}_mpnet" / "snippets.json"
            )

            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS index missing: {faiss_path}")

            if not snippets_path.exists():
                raise FileNotFoundError(f"snippets.json missing: {snippets_path}")

            # Load FAISS
            index = faiss.read_index(str(faiss_path))

            # Load snippets
            with snippets_path.open("r", encoding="utf-8") as f:
                snippets = json.load(f)

            self.indexes[name] = {
                "faiss_index": index,
                "snippets": snippets,
                "name": name,
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
