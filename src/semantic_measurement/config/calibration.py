import json
from pathlib import Path
from semantic_measurement.config.global_calibration import DATA_ROOT

def load_calibration_config():
    path = Path("data/config/threshold_calibration.json")
    with open(path, "r") as f:
        return json.load(f)
