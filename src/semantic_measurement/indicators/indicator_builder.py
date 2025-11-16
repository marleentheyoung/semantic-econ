# semantic_measurement/indicators/indicator_builder.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ParagraphHit:
    """Paragraph returned by ConceptRetriever."""
    faiss_id: int
    similarity: float
    sentence_count: int
    section: str  # 'management' or 'qa'
    call_id: str


class IndicatorBuilder:
    """
    Converts paragraph-level concept hits into call-level indicators.

    Computes:
    - Exposure: (# concept paragraphs) / (# total paragraphs in call)
    - Intensity: sum(similarity scores)
    - AvgSim: average similarity score

    All computed:
    - overall
    - management-only
    - qa-only
    """

    def __init__(self, call_metadata: Dict[str, Dict[str, Any]]):
        """
        call_metadata:
            {call_id: {
                "total_snippets",
                "management_snippets",
                "qa_snippets",
                ...
            }}
        """
        self.call_metadata = call_metadata

    def build_indicators(
        self,
        hits_by_call: Dict[str, List[ParagraphHit]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Input:
            hits_by_call = {
                call_id: [ParagraphHit, ParagraphHit, ...]
            }

        Output:
            indicators = {
                call_id: {
                    'exposure': float,
                    'avgSim': float,
                    'intensity': float,
                    'mgmt_exposure': float,
                    'qa_exposure': float,
                    ...
                }
            }
        """
        indicators: Dict[str, Dict[str, Any]] = {}

        for call_id, hits in hits_by_call.items():
            meta = self.call_metadata.get(call_id)
            if meta is None:
                # skip calls not in metadata
                continue

            total_par = meta["total_snippets"]
            mgmt_par = meta["management_snippets"]
            qa_par = meta["qa_snippets"]

            # Split hits by section
            mgmt_hits = [h for h in hits if h.section == "management"]
            qa_hits   = [h for h in hits if h.section == "qa"]

            ### ——————————————————————————————
            ### OVERALL
            ### ——————————————————————————————
            total_hits = len(hits)
            sim_scores = [h.similarity for h in hits]

            exposure = total_hits / total_par if total_par > 0 else 0
            avgSim   = sum(sim_scores)/len(sim_scores) if sim_scores else 0
            intensity = sum(sim_scores)

            ### ——————————————————————————————
            ### MANAGEMENT
            ### ——————————————————————————————
            mgmt_scores = [h.similarity for h in mgmt_hits]

            mgmt_exposure = len(mgmt_hits) / mgmt_par if mgmt_par > 0 else 0
            mgmt_avgSim   = sum(mgmt_scores)/len(mgmt_scores) if mgmt_scores else 0
            mgmt_intensity = sum(mgmt_scores)

            ### ——————————————————————————————
            ### QA
            ### ——————————————————————————————
            qa_scores = [h.similarity for h in qa_hits]

            qa_exposure = len(qa_hits) / qa_par if qa_par > 0 else 0
            qa_avgSim   = sum(qa_scores)/len(qa_scores) if qa_scores else 0
            qa_intensity = sum(qa_scores)

            indicators[call_id] = {
                # TOTALS
                "exposure": exposure,
                "avgSim": avgSim,
                "intensity": intensity,

                # MGMT
                "mgmt_exposure": mgmt_exposure,
                "mgmt_avgSim": mgmt_avgSim,
                "mgmt_intensity": mgmt_intensity,

                # QA
                "qa_exposure": qa_exposure,
                "qa_avgSim": qa_avgSim,
                "qa_intensity": qa_intensity,

                # Counters
                "n_hits_total": total_hits,
                "n_hits_mgmt": len(mgmt_hits),
                "n_hits_qa": len(qa_hits),
            }

        return indicators
