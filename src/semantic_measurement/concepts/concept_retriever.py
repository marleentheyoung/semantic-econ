from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.indicators.indicator_builder import ParagraphHit

class ConceptRetriever:
    """
    Multi-pattern concept retriever built on top of SemanticRetriever.
    Applies a similarity threshold τ and groups hits by call_id.
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        patterns: List[str],
        threshold: float,
        top_k: int = 250,
    ):
        self.retriever = retriever
        self.patterns = patterns
        self.threshold = threshold
        self.top_k = top_k

    # -------------------------------------------------------
    # Main retrieval API
    # -------------------------------------------------------
    def retrieve_hits(self) -> Dict[str, List[ParagraphHit]]:
        """
        Returns:
            dict[call_id → list[RetrievedHit]]
        """
        hits_by_call = {}

        for pattern in self.patterns:
            results = self.retriever.search_by_text(pattern, top_k=self.top_k)

            for item in results:
                if item["score"] < self.threshold:
                    continue

                snippet = item["snippet"]

                hit = ParagraphHit(
                    faiss_id=item["faiss_id"],
                    similarity=item["score"],
                    sentence_count=snippet.get("sentence_count", 1),
                    section=snippet.get("section", "management"),
                    call_id=snippet["call_id"],
                )

                hits_by_call.setdefault(hit.call_id, []).append(hit)

        return hits_by_call
