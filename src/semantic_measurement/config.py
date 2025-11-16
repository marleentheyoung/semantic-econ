# src/semantic_measurement/config.py

from pathlib import Path

# Project root = two levels up from this file: src/semantic_measurement/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base data directory.
# Default: <project_root>/data
# You can change this to an absolute path if you prefer.
DATA_ROOT = PROJECT_ROOT / "data"
