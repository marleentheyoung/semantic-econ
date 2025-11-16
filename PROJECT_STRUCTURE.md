# PROJECT_STRUCTURE.md

This document describes the expected file and module layout for the semantic-search economic measurement project.
It reflects the **updated preprocessing pipeline**, including PDF metadata parsing, transcript extraction, structured transcript building, and paragraph extraction.

The structure separates **reusable library code (`src/`)** from **executable pipeline steps (`scripts/`)** and **experiment configuration (`configs/`)**.

---

## Top-level layout

```
semantic-measurement/
├─ src/
│  └─ semantic_measurement/
│     ├─ data/
│     ├─ preprocessing/
│     ├─ embeddings/
│     ├─ index/
│     ├─ queries/
│     ├─ retrieval/
│     ├─ validation/
│     ├─ indicators/
│     ├─ econometrics/
│     └─ utils/
├─ scripts/
├─ configs/
├─ notebooks/
├─ tests/
├─ pyproject.toml
├─ README.md
└─ PROJECT_STRUCTURE.md
```

---

# Directory overview

## `src/semantic_measurement/`

All reusable code lives here.
No scripts, no CLI, no hardcoded I/O paths — only importable functions and classes.

---

## `data/` — PDF → transcripts → structured transcripts

Contains modules that read raw files, extract transcripts, handle metadata, manage streaming, and output structured transcript JSON.

```
src/semantic_measurement/data/
├─ pdf_metadata.py             # extract company/ticker/date from PDF first pages
├─ pdf_processing.py           # memory-efficient transcript extraction from PDFs
├─ structure_transcripts.py    # build structured transcripts (speaker segments + metadata)
├─ streaming_json.py           # safe iteration/writing of huge JSON arrays
└─ schemas.py                  # TranscriptRecord, SpeakerSegment, StructuredTranscript
```

**Key roles:**

* **`pdf_metadata.py`**
  Parses first pages of PDFs to extract firm identifiers, dates, region info.

* **`pdf_processing.py`**
  Reads PDF text in memory-safe batches and writes `transcripts_data_part*.json`.

* **`structure_transcripts.py`**
  Converts raw transcript JSON parts into structured JSON:

  * remove FactSet metadata
  * split into Management/Q&A sections
  * speaker segmentation
  * paragraph lists
  * attach metadata from `pdf_metadata`

* **`streaming_json.py`**
  Provides:

  * `stream_json_array`
  * `process_json_in_chunks`
  * `write_json_streaming`
  * `MemoryEfficientJSONProcessor`

These enable processing multi-GB transcript datasets without RAM issues.

---

## `preprocessing/` — text cleaning, segmentation, paragraph extraction

```
src/semantic_measurement/preprocessing/
├─ cleaning.py                # unicode normalization, whitespace, boilerplate removal
├─ filters.py                 # remove FactSet disclaimers, filter empty/noisy segments
├─ segmentation.py            # split speakers, paragraphs, mgmt/Q&A identification
├─ paragraph_extractor.py     # flatten structured transcripts → paragraph-level records
└─ pipeline.py                # PreprocessingPipeline (wrapper for text-only workflows)
```

**Key components:**

* **`filters.py`**
  Contains `remove_factset_metadata` and similar cleaning rules.

* **`segmentation.py`**
  Includes `split_and_extract_speakers` and supporting parsing logic.

* **`paragraph_extractor.py`**
  Converts structured transcripts into embedding-ready paragraph dictionaries.

* **`pipeline.py`**
  Provides a high-level text preprocessing wrapper (useful for downstream modules).

---

## `embeddings/`

Text encoder backends and embedding utilities.

```
src/semantic_measurement/embeddings/
├─ base.py
├─ sentence_transformers_backend.py
├─ cache.py
└─ pooling.py
```

---

## `index/`

Approximate nearest-neighbor search (FAISS) + metadata mapping.

```
src/semantic_measurement/index/
├─ faiss_index.py
├─ metadata_store.py
├─ builders.py
└─ persistence.py
```

---

## `queries/`

Concept definitions, TCFD-style query patterns, and optional LLM-assisted expansion.

```
src/semantic_measurement/queries/
├─ registry.py
├─ concepts/
└─ expansion.py
```

---

## `retrieval/`

Similarity scoring, thresholding, and concept-level retrieval.

```
src/semantic_measurement/retrieval/
├─ similarity.py
├─ scorer.py
├─ thresholds.py
├─ retrieval.py
└─ api.py
```

---

## `validation/`

Human/LLM annotation workflows, ROC/PR metrics, and τ threshold estimation.

```
src/semantic_measurement/validation/
├─ sampling.py
├─ annotation_schema.py
├─ llm_labeler.py
├─ metrics.py
└─ threshold_search.py
```

---

## `indicators/`

Compute exposure, similarity-based metrics, and panel-level topic indicators.

```
src/semantic_measurement/indicators/
├─ exposure.py
├─ similarity_stats.py
├─ intensity.py
└─ panel_builder.py
```

---

## `econometrics/`

Downstream regressions, tables, and robustness checks.

```
src/semantic_measurement/econometrics/
├─ controls.py
├─ models.py
└─ robustness.py
```

---

## `utils/`

Lightweight helpers shared across modules.

```
src/semantic_measurement/utils/
├─ pdf_reader.py       # optional: a wrapper for the PDF reader
├─ logging.py
├─ ids.py
└─ timing.py
```

---

# `scripts/` — runnable pipeline steps

Each script performs exactly one stage of the pipeline by calling code from `src/semantic_measurement`.

```
scripts/
├─ extract_transcripts.py        # PDFs → transcripts_data_part*.json
├─ structure_transcripts.py      # transcript parts → structured transcripts
├─ extract_paragraphs.py         # structured transcripts → paragraph dataset
├─ embed_segments.py
├─ build_index.py
├─ validate_concept.py
├─ build_indicators.py
└─ run_full_pipeline.py
```

Scripts should follow this pattern:

1. Parse CLI arguments
2. Load raw/intermediate data
3. Call the appropriate function/class from `src/`
4. Save outputs

---

# `configs/` — YAML experiment and pipeline configuration

```
configs/
├─ pdf_processing.yaml
├─ preprocessing.yaml
├─ embedding.yaml
├─ faiss_index.yaml
└─ concepts.yaml
```

---

# `notebooks/`

Exploration notebooks and figure/table generation for research outputs.

---

# `tests/`

One test file per module, mirroring the structure under `src/`.

```
tests/
├─ test_data.py
├─ test_preprocessing.py
├─ test_embeddings.py
├─ test_index.py
├─ test_retrieval.py
├─ test_indicators.py
└─ test_econometrics.py
```