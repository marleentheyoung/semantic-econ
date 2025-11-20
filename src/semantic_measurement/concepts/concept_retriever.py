from __future__ import annotations
from typing import List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
from tqdm import tqdm

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

    def retrieve_hits_streaming(self) -> Iterator[Tuple[str, List[ParagraphHit]]]:
        """
        Generator that yields (call_id, hits) pairs after processing all patterns.
        Memory-efficient alternative to retrieve_hits().
        
        Yields:
            (call_id, list[ParagraphHit]) tuples
        """
        hits_by_call: Dict[str, List[ParagraphHit]] = {}

        for pattern in tqdm(self.patterns, desc="Retrieving patterns", unit="pattern"):
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

        # Yield results one call at a time
        for call_id, hits in hits_by_call.items():
            yield call_id, hits


    def retrieve_hits_batched(self, batch_size: int = 1000) -> Iterator[Dict[str, List[ParagraphHit]]]:
        """
        Generator that yields batches of hits for better memory control.
        
        Parameters:
            batch_size: Number of calls per batch
            
        Yields:
            dict[call_id → list[ParagraphHit]] batches
        """
        batch: Dict[str, List[ParagraphHit]] = {}

        for call_id, hits in self.retrieve_hits_streaming():
            batch[call_id] = hits

            if len(batch) >= batch_size:
                yield batch
                batch = {}

        if batch:
            yield batch