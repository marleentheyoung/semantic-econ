# 🧩 **Migration TODO List**

## ✅ Legend

* ✔️🙂 = finished
* ⬜ = still to do
* 🔧 = pending review / optional improvement

---

# 1. **Embedding & Index Infrastructure (done)**

### ✔️🙂 Embedding backend (SentenceTransformerBackend)

### ✔️🙂 Embedding script (`embed_segments.py`)

### ✔️🙂 Chunked `.npy` storage

### ✔️🙂 FAISS index builder (`build_index_from_embeddings.py`)

### ✔️🙂 Two-index architecture (SP500 + STOXX600)

### ✔️🙂 Stable ordering of `snippets.json`

---

# 2. **Metadata & Paragraph Layer**

### ✔️🙂 ParagraphRecord schema fixed

### ✔️🙂 Sentence count utility implemented

### ✔️🙂 Paragraph extraction pipeline aligned

### ⬜ Move call-level metadata into:

```
data/metadata/SP500_calls.json
data/metadata/STOXX600_calls.json
```

### ⬜ Create helper loader for call metadata

### ⬜ Verify each snippet has pointer to call_id for aggregation

---

# 3. **Retrieval Layer (new architecture)**

### ✔️🙂 FAISS-based retrieval design approved

### ✔️🙂 Region-separated vector stores

### ✔️🙂 Query-side embedding → vector

### ✔️🙂 Merge-US-and-EU logic implemented

### ⬜ Rename class to simpler name

**Options:**

* `SemanticRetriever` (recommended)
* `Retriever`
* `ConceptRetriever` (redundant if used for next layer)
* `UnifiedRetriever`

➡️ **I recommend: `SemanticRetriever`**
(short, clear, not intimidating, describes exactly what it is)

### ⬜ Apply renaming across module

### ⬜ Add `search_by_embedding()` helper for ROC validation use

### ⬜ Add unit tests for top-k consistency

---

# 4. **Concept Retrieval Layer (multi-query)**

### ⬜ Create folder: `semantic_measurement/concepts/` (or use existing retrieval/)

### ⬜ Implement `ConceptRetriever` class

* loads queries
* gets raw paragraph matches from SemanticRetriever
* applies threshold τc
* stores similarity + metadata

### ⬜ Implement multi-pattern retrieval for each concept

### ⬜ Add mgmt/QA split option

---

# 5. **Indicators Layer**

### ✔️🙂 ExposureBuilder base version

### ⬜ Extend ExposureBuilder for three measures:

* Exposure
* AvgSim
* Intensity

### ⬜ Implement LaTeX-aligned formulas

### ⬜ Add support for per-call denominator (# paragraphs)

### ⬜ Support segment-type splits (mgmt/qa)

### ⬜ Add sanity-check tests

---

# 6. **Panel Construction Layer (pipeline/)**

### ⬜ Create folder: `semantic_measurement/pipeline/`

### ⬜ Implement `panel_builder.py`

* merge exposure results with call metadata
* assign firm-year and firm-quarter keys
* handle dual-index region tagging

### ⬜ Add optional lag generation:

```
--lag 1, --lag 4, --ma 2
```

### ⬜ Implement clean output writer to `.parquet`

---

# 7. **Batch Runner**

### ⬜ Write `scripts/run_topic_panel.py` (single topic)

### ⬜ Write `scripts/run_batch_topics.py` (all topics)

### ⬜ Remove all user prompts present in old scripts

### ⬜ Add CLI arguments:

* `--topic`
* `--run-roc`
* `--lags`
* `--max-k`

### ⬜ Capture logs in `outputs/logs/`

---

# 8. **Validation Layer**

### ⬜ Integrate SimpleROCValidator with new retrieval

### ⬜ Add flag to recompute or reuse thresholds

### ⬜ Add sampling utility for manual relevance checks

### ⬜ Add AUC + threshold plots (saved to outputs/)

---

# 9. **Documentation & Final Cleanup**

### ✔️🙂 Methodology section matches architecture (minor edits pending)

### ⬜ Update README for entire project

### ⬜ Add architecture diagram (I can generate this)

### ⬜ Add example notebook:

```
notebooks/demo_topic_retrieval.ipynb
```

### ⬜ Remove all old index & search code

### ⬜ Add tests for:

* retrieval top-k correctness
* exposure & intensity aggregation
* panel builder alignment

---

# ✨ Notes on the Rename of DualIndexRetriever

Your instinct was correct — **the name sounds more complicated than the logic**.

We have a few options:

### Most natural + simple:

**`SemanticRetriever`**

* short
* describes exactly what it is
* avoids detail about number of indices

### If we want to be explicit but not scary:

**`UnifiedRetriever`**

* means “I unify multiple vector stores into one search interface”
* accurate but still simple

### If we want to stay minimal:

**`Retriever`**

* short, clean, but a bit generic

---

# 👍 Recommended Rename Decision

### → **Rename `DualIndexRetriever` to `SemanticRetriever`.**

The class will still internally load two FAISS indices but the user doesn’t need to know that.

---

If you want, I can now:

* implement the rename,
* update the class docstring and folder structure,
* or proceed with the next block of code (ConceptRetriever / Indicators / Panel builder).

Just tell me.
