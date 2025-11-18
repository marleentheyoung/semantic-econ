from semantic_measurement.indicators.indicator_builder import IndicatorBuilder

# Fake one-call metadata
meta = {
    "CALL123": {
        "total_snippets": 100,
        "total_sentences" : 300,
        "management_sentences" : 180,
        "qa_sentences" : 120,
        "management_snippets": 60,
        "qa_snippets": 40,
    }
}

# Fake hits
from semantic_measurement.indicators.indicator_builder import ParagraphHit
hits = {
    "CALL123": [
        ParagraphHit(faiss_id=1, similarity=0.60, sentence_count=4, section="management", call_id="CALL123"),
        ParagraphHit(faiss_id=2, similarity=0.55, sentence_count=3, section="qa", call_id="CALL123"),
    ]
}

ib = IndicatorBuilder(meta)
out = ib.build_indicators(hits)

print(out["CALL123"])
