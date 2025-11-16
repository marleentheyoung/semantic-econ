from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever


@dataclass
class RetrievedHit:
    score: float
    call_id: str
    index_name: str
    snippet: dict


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
    def retrieve_hits(self) -> Dict[str, List[RetrievedHit]]:
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
                call_id = snippet["call_id"]

                hit = RetrievedHit(
                    score=item["score"],
                    call_id=call_id,
                    index_name=item["index_name"],
                    snippet=snippet,
                )

                if call_id not in hits_by_call:
                    hits_by_call[call_id] = []
                hits_by_call[call_id].append(hit)

        return hits_by_call
