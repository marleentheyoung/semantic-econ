# src/semantic_measurement/config.py

"""
Global configuration for the semantic measurement package.

This file centralizes all paths, model names, and defaults so that
scripts and modules do not hard-code environment-specific values.

You can safely modify values here without touching other code.
"""

from pathlib import Path


# ----------------------------------------------------------------------
# Project Paths
# ----------------------------------------------------------------------

# PROJECT_ROOT = repository root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Base data directory: <project_root>/data
# Override this if your dataset lives elsewhere.
DATA_ROOT = PROJECT_ROOT / "data"


# ----------------------------------------------------------------------
# Embedding Models
# ----------------------------------------------------------------------

# Default embedding model for semantic retrieval
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ----------------------------------------------------------------------
# Index & Embeddings Directories
# ----------------------------------------------------------------------

# Location of FAISS indexes (SP500, STOXX600)
DEFAULT_INDEX_DIR = DATA_ROOT / "indexes"

# Location of snippet embeddings
DEFAULT_EMBEDDINGS_DIR = DATA_ROOT / "embeddings"

# Which index names exist under DEFAULT_INDEX_DIR
DEFAULT_INDEX_NAMES = ["SP500", "STOXX600"]


# ----------------------------------------------------------------------
# LLM Settings (for threshold calibration / labeling)
# ----------------------------------------------------------------------

# Default Claude model for LLM labeling
# "latest" avoids 404 errors when Anthropic updates versions.
LLM_MODEL = "claude-sonnet-4-5-20250929"
