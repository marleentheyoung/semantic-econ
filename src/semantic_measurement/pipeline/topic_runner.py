# src/semantic_measurement/pipeline/topic_runner.py

from pathlib import Path
import json

from semantic_measurement.retrieval.semantic_retriever import SemanticRetriever
from semantic_measurement.concepts.concept_retriever import ConceptRetriever
from semantic_measurement.indicators.indicator_builder import IndicatorBuilder
from semantic_measurement.embeddings.sentence_transformers_backend import SentenceTransformerBackend
from semantic_measurement.pipeline.panel_builder import PanelBuilder
from semantic_measurement.config import DATA_ROOT, DEFAULT_EMBEDDING_MODEL


class TopicRunner:

    def __init__(
        self,
        call_metadata,
        queries_root: Path | None = None,
        index_dir: Path | None = None,
        embeddings_dir: Path | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        # Base data directory
        self.data_root = DATA_ROOT

        # Defaults OR overrides
        self.queries_root = Path(queries_root) if queries_root else self.data_root / "queries"
        self.index_dir = Path(index_dir) if index_dir else self.data_root / "indexes"
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else self.data_root / "embeddings"

        self.index_names = ["SP500", "STOXX600"]
        self.model_name = model_name
        self.call_metadata = call_metadata

    # ---------------------------------------------------------
    # Load queries, threshold, description for a topic
    # ---------------------------------------------------------
    def load_topic_config(self, topic_name: str):
        query_file = self.queries_root / f"{topic_name}.json"
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")

        with query_file.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        queries = cfg["queries"]
        threshold = cfg.get("threshold", 0.42)
        description = cfg.get("description", topic_name)

        return queries, threshold, description

    # ---------------------------------------------------------
    # Run full pipeline for a topic
    # ---------------------------------------------------------
    def run_topic(self, topic_name: str):

        queries, threshold, description = self.load_topic_config(topic_name)

        # Embedder
        embedder = SentenceTransformerBackend(self.model_name)

        # Semantic retrieval
        retriever = SemanticRetriever(
            embedder=embedder,
            index_dir=self.index_dir,
            embeddings_dir=self.embeddings_dir,
            index_names=self.index_names,
        )

        concept = ConceptRetriever(
            retriever,      # SemanticRetriever instance
            queries,        # list of patterns
            threshold,      # float
        )

        hits = concept.retrieve_hits()

        # Indicator builder
        indicators = IndicatorBuilder(self.call_metadata).build_indicators(hits)

        # Final panel
        panel = PanelBuilder(self.call_metadata).build(indicators)
        panel["topic"] = topic_name
        panel["topic_description"] = description

        return panel
