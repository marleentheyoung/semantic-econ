# src/semantic_measurement/pipeline/panel_builder.py

from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class PanelBuilder:
    """
    Build a firm-quarter panel by merging:
      1. call metadata  (per call_id)
      2. computed indicators (per call_id)
    """

    call_metadata: Dict[str, dict]

    # ------------------------------------------------------------------
    def metadata_to_df(self) -> pd.DataFrame:
        """Convert call metadata dict → DataFrame indexed by call_id."""
        meta_df = pd.DataFrame.from_dict(self.call_metadata, orient="index")

        # Ensure canonical fields exist
        required_cols = ["ticker", "year", "quarter"]
        missing = [c for c in required_cols if c not in meta_df.columns]
        if missing:
            raise KeyError(f"Missing metadata fields: {missing}")

        meta_df.index.name = "call_id"
        return meta_df

    # ------------------------------------------------------------------
    def build(self, indicators: Dict[str, Dict[str, float]]) -> pd.DataFrame:
      meta_df = self.metadata_to_df()

      # Convert indicator dict → DataFrame indexed by call_id
      indicators_df = pd.DataFrame.from_dict(indicators, orient="index")

      # Merge metadata + indicators
      panel = meta_df.join(indicators_df, how="left")

      panel = panel.sort_values(["ticker", "year", "quarter"]).reset_index(drop=True)
      panel["quarter_id"] = panel["year"].astype(str) + panel["quarter"].astype(str)

      return panel

