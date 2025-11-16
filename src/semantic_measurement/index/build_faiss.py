# src/semantic_measurement/index/build_faiss.py

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Literal, Dict, Any
import numpy as np
import faiss
from tqdm import tqdm

from .faiss_index import save_index


Metric = Literal["ip", "l2"]


def load_chunk_paths(chunks_index_file: Path) -> List[Path]:
    """Load the list of chunk .npy files from chunks_index.json."""
    with chunks_index_file.open("r", encoding="utf-8") as f:
        paths = json.load(f)
    return [Path(p) for p in paths]


def build_index_from_embeddings(
    chunks_index_file: Path,
    snippets_file: Path,
    output_dir: Path,
    index_name: str = "semantic_index",
    metric: Metric = "ip",
) -> Dict[str, Any]:
    """
    Build a FAISS index from stored embedding chunks.

    Parameters
    ----------
    chunks_index_file : Path
        JSON file listing embedding chunk paths.
    snippets_file : Path
        JSON file with paragraph/snippet metadata (ordered).
    output_dir : Path
        Directory where index and metadata will be saved.
    index_name : str
        Name for the index.
    metric : {"ip", "l2"}
        Metric for the FAISS index.

    Returns
    -------
    metadata : dict
        Index metadata including dimension and ntotal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load chunk paths
    # -----------------------------------------------------
    chunk_paths = load_chunk_paths(chunks_index_file)
    if not chunk_paths:
        raise ValueError(f"No chunk files listed in {chunks_index_file}")

    # -----------------------------------------------------
    # Determine embedding dimension from first chunk
    # -----------------------------------------------------
    first_chunk = np.load(chunk_paths[0])
    dim = first_chunk.shape[1]

    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    # -----------------------------------------------------
    # Add first chunk
    # -----------------------------------------------------
    index.add(first_chunk.astype(np.float32))
    del first_chunk

    # -----------------------------------------------------
    # Add remaining chunks
    # -----------------------------------------------------
    for path in tqdm(chunk_paths[1:], desc="Adding chunks to FAISS"):
        emb = np.load(path).astype(np.float32)
        index.add(emb)
        del emb

    # -----------------------------------------------------
    # Save FAISS index
    # -----------------------------------------------------
    index_path = output_dir / f"{index_name}.faiss"
    save_index(index, index_path)

    # -----------------------------------------------------
    # Write metadata
    # -----------------------------------------------------
    metadata = {
        "index_name": index_name,
        "dimension": dim,
        "metric": metric,
        "ntotal": index.ntotal,
        "files": {
            "index": str(index_path),
            "snippets": str(snippets_file),
            "chunks_index": str(chunks_index_file),
            "metadata": str(output_dir / f"{index_name}_metadata.json"),
        },
    }

    metadata_path = output_dir / f"{index_name}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
