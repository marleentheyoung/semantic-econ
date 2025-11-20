from pathlib import Path
import json

from semantic_measurement.config.global_calibration import DATA_ROOT

def load_queries(topic: str):
    query_file = Path("data/queries") / f"{topic}.json"
    with open(query_file, "r") as f:
        return json.load(f), query_file

def load_all_snippets():
    sp = DATA_ROOT / "embeddings" / "SP500_mpnet" / "snippets.json"
    eu = DATA_ROOT / "embeddings" / "STOXX600_mpnet" / "snippets.json"

    with sp.open("r", encoding="utf-8") as f:
        sp_snippets = json.load(f)
    with eu.open("r", encoding="utf-8") as f:
        eu_snippets = json.load(f)

    return sp_snippets + eu_snippets
