# src/semantic_measurement/index/faiss_index.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import faiss


Metric = Literal["ip", "l2"]


@dataclass
class FaissIndexInfo:
    """
    Lightweight metadata about a FAISS index, useful for logging / checks.
    """
    dimension: int
    metric: Metric
    ntotal: int


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    """
    Ensure the array is float32 and C-contiguous, as FAISS expects.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def create_flat_index(dimension: int, metric: Metric = "ip") -> faiss.Index:
    """
    Create a simple flat FAISS index.

    - metric="ip": inner product (use with normalized embeddings for cosine)
    - metric="l2": L2 Euclidean distance
    """
    if metric == "ip":
        return faiss.IndexFlatIP(dimension)
    elif metric == "l2":
        return faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def add_embeddings(index: faiss.Index, embeddings: np.ndarray) -> None:
    """
    Add a batch of embeddings to an existing index.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    embeddings = _ensure_float32(embeddings)
    index.add(embeddings)


def search(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic search wrapper.

    Returns:
    - distances: shape (n_queries, k)
    - indices:   shape (n_queries, k)
    """
    if query_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {query_embeddings.shape}")
    queries = _ensure_float32(query_embeddings)
    distances, indices = index.search(queries, k)
    return distances, indices


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Save a FAISS index to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    """
    Load a FAISS index from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))


def get_index_info(index: faiss.Index, metric: Metric = "ip") -> FaissIndexInfo:
    """
    Extract basic index metadata for logging and sanity checks.
    """
    dimension = index.d
    ntotal = index.ntotal
    return FaissIndexInfo(
        dimension=dimension,
        metric=metric,
        ntotal=ntotal,
    )
