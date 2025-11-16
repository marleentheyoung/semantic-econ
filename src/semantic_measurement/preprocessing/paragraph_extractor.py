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

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text if len(text) >= 5 else ""


# ------------------------------------------------------------
# Extract paragraphs from segments
# ------------------------------------------------------------
def _iter_paragraphs_from_segments(
    transcript: Dict[str, Any],
    json_file: Path,
    region: str,
    segments_key: str,
    section_label: str,
) -> Iterable[Dict[str, Any]]:

    segments = transcript.get(segments_key) or []

    for seg in segments:
        speaker = seg.get("speaker") or "Unknown"
        profession = seg.get("profession") or None
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
                "section": section_label,
                "sentence_count": count_sentences(cleaned),
                "call_id": transcript.get("file", transcript.get("filename", "")),
                "region": region,
                "source_file": json_file.name,
            }


def extract_paragraphs_from_transcript(
    transcript: Dict[str, Any],
    json_file: Path,
    region: str,
) -> List[Dict[str, Any]]:

    recs = []

    recs.extend(
        _iter_paragraphs_from_segments(
            transcript,
            json_file=json_file,
            region=region,
            segments_key="speaker_segments_management",
            section_label="management",
        )
    )

    recs.extend(
        _iter_paragraphs_from_segments(
            transcript,
            json_file=json_file,
            region=region,
            segments_key="speaker_segments_qa",
            section_label="qa",
        )
    )

    return recs


# ------------------------------------------------------------
# Folder-level entry point
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
        f"✅ Finished: {total_calls} calls → {total_paragraphs} paragraphs written to {output_path}"
    )
