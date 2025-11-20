"""
Paragraph-level extraction from structured transcripts.
Produces JSONL where each line is a clean ParagraphRecord-compatible dict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, List
from tqdm import tqdm

# If you already have this utility somewhere, import it:
from semantic_measurement.utils.text import count_sentences


# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------
def clean_paragraph_text(text: str) -> str:
    if not text:
        return ""

    # Normalize whitespace
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # ------------------------------
    # Remove CallStreet noise blocks
    # ------------------------------
    lower = text.lower()

    # Common footer/header markers
    if "www.callstreet.com" in lower:
        return ""
    if "copyright" in lower and "callstreet" in lower:
        return ""
    if "212-849-4070" in lower:
        return ""
    if "©" in text and "callstreet" in lower:
        return ""

    # Throw away paragraphs that are only symbols or <= 5 chars
    if len(text) < 5:
        return ""

    return text


# ------------------------------------------------------------
# Emit management paragraphs (unchanged)
# ------------------------------------------------------------
def _iter_management_paragraphs(
    transcript: Dict[str, Any],
    json_file: Path,
    region: str,
) -> Iterable[Dict[str, Any]]:

    segments = transcript.get("speaker_segments_management") or []

    for seg in segments:
        speaker = seg.get("speaker") or "Unknown"
        profession = seg.get("profession")
        seg_paragraphs = seg.get("paragraphs") or []

        for raw_para in seg_paragraphs:
            cleaned = clean_paragraph_text(raw_para)
            if not cleaned:
                continue

            yield {
                "text": cleaned,
                "company_name": transcript.get("company_name"),
                "ticker": transcript.get("ticker"),
                "date": transcript.get("date"),
                "quarter": transcript.get("quarter"),
                "year": transcript.get("year"),
                "speaker": speaker,
                "profession": profession,
                "section": "management",
                "sentence_count": count_sentences(cleaned),
                "call_id": transcript.get("call_id"),
                "region": region,
                "source_file": json_file.name,
            }


# ------------------------------------------------------------
# NEW: Q+A pairing logic
# ------------------------------------------------------------
def _iter_qa_pairs(
    transcript: Dict[str, Any],
    json_file: Path,
    region: str,
) -> Iterable[Dict[str, Any]]:

    segments = transcript.get("speaker_segments_qa") or []

    if not segments:
        return

    i = 0
    n = len(segments)

    while i < n:

        seg = segments[i]
        qa_type = seg.get("qa")
        speaker_q = None
        question_text = None

        # ---------------------------------------
        # 1. Detect a QUESTION segment
        # ---------------------------------------
        if qa_type == "Q":
            speaker_q = seg.get("speaker") or "Unknown"
            q_paragraphs = seg.get("paragraphs") or []
            q_clean = " ".join(clean_paragraph_text(p) for p in q_paragraphs if clean_paragraph_text(p))
            question_text = q_clean
            i += 1
        else:
            # skip standalone A segments until first Q appears
            i += 1
            continue

        # ---------------------------------------
        # 2. Collect ANSWER segment(s)
        #    All consecutive A segments until next Q
        # ---------------------------------------
        answer_parts = []
        speaker_a = None

        while i < n and segments[i].get("qa") == "A":
            a_seg = segments[i]
            if speaker_a is None:
                speaker_a = a_seg.get("speaker") or "Management"

            a_paragraphs = a_seg.get("paragraphs") or []
            for p in a_paragraphs:
                cleaned = clean_paragraph_text(p)
                if cleaned:
                    answer_parts.append(cleaned)

            i += 1

        answer_text = " ".join(answer_parts).strip()

        if not question_text or not answer_text:
            # Skip degenerate pairs
            continue

        combined = f"Q: {question_text}\nA: {answer_text}"
        combined_clean = combined.strip()

        yield {
            "text": combined_clean,
            "question": question_text,
            "answer": answer_text,
            "speaker_q": speaker_q,
            "speaker_a": speaker_a,
            "company_name": transcript.get("company_name"),
            "ticker": transcript.get("ticker"),
            "date": transcript.get("date"),
            "quarter": transcript.get("quarter"),
            "year": transcript.get("year"),
            "section": "qa_pair",
            "sentence_count": count_sentences(combined_clean),
            "call_id": transcript.get("call_id"),
            "region": region,
            "source_file": json_file.name,
        }


# ------------------------------------------------------------
# Public API: extract paragraphs from transcript
# ------------------------------------------------------------
def extract_paragraphs_from_transcript(
    transcript: Dict[str, Any],
    json_file: Path,
    region: str,
) -> List[Dict[str, Any]]:

    recs = []

    # 1. Management paragraphs unchanged
    recs.extend(
        _iter_management_paragraphs(
            transcript,
            json_file=json_file,
            region=region,
        )
    )

    # 2. Skip individual QA paragraphs entirely
    # 3. Add paired Q+A records
    recs.extend(
        _iter_qa_pairs(
            transcript,
            json_file=json_file,
            region=region,
        )
    )

    return recs


# ------------------------------------------------------------
# Folder-level entry point (unchanged)
# ------------------------------------------------------------
def extract_paragraphs_from_folder(input_folder: str, output_file: str) -> None:

    input_path = Path(input_folder)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")

    json_files = sorted(input_path.glob("structured_calls_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No structured_calls_*.json files found in {input_path}")

    # infer region
    folder_str = str(input_path).upper()
    if "SP500" in folder_str:
        region = "US"
    elif "STOXX600" in folder_str:
        region = "EU"
    else:
        region = "UNKNOWN"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_paragraphs = 0
    total_calls = 0

    with output_path.open("w", encoding="utf-8") as out_f:

        for json_file in tqdm(json_files, desc="Files", unit="file"):

            try:
                structured_calls = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"⚠️ Could not load {json_file}: {e}")
                continue

            if not isinstance(structured_calls, list):
                print(f"⚠️ Skipping {json_file}: not a list of transcripts")
                continue

            for call in structured_calls:
                total_calls += 1

                for rec in extract_paragraphs_from_transcript(
                    call,
                    json_file=json_file,
                    region=region,
                ):
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_paragraphs += 1

    print(
        f"✅ Finished: {total_calls} calls → {total_paragraphs} records written to {output_path}"
    )
