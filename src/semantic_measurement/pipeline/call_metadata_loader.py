# src/semantic_measurement/pipeline/call_metadata_loader.py

import json
from pathlib import Path
from typing import Dict


def load_call_metadata(path: Path) -> Dict[str, dict]:
    """
    Load call-level metadata from a JSON file.

    Expected format:
        {
            "AAPL_2020_Q1": { ... },
            "MSFT_2021_Q3": { ... }
        }

    Parameters
    ----------
    path : Path
        Path to metadata file (e.g. data/metadata/SP500_calls.json)

    Returns
    -------
    Dict[str, dict]
    """
    if not path.exists():
        raise FileNotFoundError(f"Call metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # ensure dict[str -> dict]
    if not isinstance(data, dict):
        raise ValueError(f"Call metadata file malformed: {path}")

    return data
