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

    ALL exposure measures are denominated in SENTENCES:

        Exposure     = (sum sentence_count of hits) / (total sentences in call)
        AvgSim       = average similarity of hits
        Intensity    = Exposure * AvgSim

    Computed overall and separately for:
        - Management-only
        - QA-only
    """

    def __init__(self, call_metadata: Dict[str, Dict[str, Any]]):
        """
        call_metadata:
            {
                call_id: {
                    "total_sentences",
                    "management_sentences",
                    "qa_sentences",
                    ...
                }
            }
        """
        self.call_metadata = call_metadata

    def build_indicators(
        self,
        hits_by_call: Dict[str, List[ParagraphHit]],
    ) -> Dict[str, Dict[str, Any]]:

        indicators: Dict[str, Dict[str, Any]] = {}

        for call_id, hits in hits_by_call.items():
            meta = self.call_metadata.get(call_id)
            if meta is None:
                continue

            total_sent = meta["total_sentences"]
            mgmt_sent = meta["management_sentences"]
            qa_sent   = meta["qa_sentences"]

            # Split hits
            mgmt_hits = [h for h in hits if h.section == "management"]
            qa_hits   = [h for h in hits if h.section == "qa"]

            ### -------------------------------------------------
            ### OVERALL
            ### -------------------------------------------------
            total_hit_sent = sum(h.sentence_count for h in hits)
            sim_scores = [h.similarity for h in hits]

            exposure = total_hit_sent / total_sent if total_sent > 0 else 0
            avgSim   = sum(sim_scores)/len(sim_scores) if sim_scores else 0
            intensity = exposure * avgSim

            ### -------------------------------------------------
            ### MANAGEMENT
            ### -------------------------------------------------
            mgmt_hit_sent = sum(h.sentence_count for h in mgmt_hits)
            mgmt_scores = [h.similarity for h in mgmt_hits]

            mgmt_exposure = mgmt_hit_sent / mgmt_sent if mgmt_sent > 0 else 0
            mgmt_avgSim   = sum(mgmt_scores)/len(mgmt_scores) if mgmt_scores else 0
            mgmt_intensity = mgmt_exposure * mgmt_avgSim

            ### -------------------------------------------------
            ### QA
            ### -------------------------------------------------
            qa_hit_sent = sum(h.sentence_count for h in qa_hits)
            qa_scores = [h.similarity for h in qa_hits]

            qa_exposure = qa_hit_sent / qa_sent if qa_sent > 0 else 0
            qa_avgSim   = sum(qa_scores)/len(qa_scores) if qa_scores else 0
            qa_intensity = qa_exposure * qa_avgSim

            indicators[call_id] = {
                # TOTAL
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

                # Counters (for debugging)
                "n_hits_total": len(hits),
                "n_hits_mgmt": len(mgmt_hits),
                "n_hits_qa": len(qa_hits),

                # Sentence-weighted counts
                "sentences_total_hits": total_hit_sent,
                "sentences_mgmt_hits": mgmt_hit_sent,
                "sentences_qa_hits": qa_hit_sent,
            }

        return indicators
