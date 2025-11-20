import re
import pandas as pd


# ===============================================================
# 1. Detect which Q&A format we are dealing with
# ===============================================================

def detect_format(text):
    if re.search(r'\.{10,}', text):   # long dotted lines → CallStreet
        return "callstreet"
    if re.search(r'<[QA]\s*[–-]\s*[^>]+>', text):   # <Q – Name> inline tags
        return "inline_qna"
    return "unknown"



# ===============================================================
# 2. INLINE Q&A FORMAT
#    Example: <Q – Scott Davis>: Question text...
# ===============================================================

INLINE_PATTERN = re.compile(
    r'(?P<tag><(?P<qa>[QA])\s*[–-]\s*(?P<name>[^>]+)>:?)[ \t]*'
)

def parse_inline_qna(text):
    elements = []
    matches = list(INLINE_PATTERN.finditer(text))

    if not matches:
        return []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)

        qa = m.group("qa")
        speaker = m.group("name").strip()

        segment_text = text[start:end].strip()

        paragraphs = [
            p.replace("\n", " ").strip()
            for p in re.split(r'\n\s*\n', segment_text)
            if p.strip()
        ]

        elements.append({
            "speaker": speaker,
            "profession": None,
            "qa": qa,
            "paragraphs": paragraphs
        })

    return elements



# ===============================================================
# 3. CALLSTREET FORMAT (with dotted separators)
#    Structure:
#       Name
#       Profession
#       Q/A line OR paragraph starting with Q/A
# ===============================================================

def extract_qa_flag_and_paragraphs(lines):
    qa = None

    # line index 2 is the Q/A marker in most transcripts
    if len(lines) >= 3:
        line = lines[2].strip()

        if line in ("Q", "A"):
            qa = line
            paragraphs = lines[3:]
            return qa, paragraphs

        if line.startswith("Q "):
            qa = "Q"
            paragraphs = [line[2:].strip()] + lines[3:]
            return qa, paragraphs

        if line.startswith("A "):
            qa = "A"
            paragraphs = [line[2:].strip()] + lines[3:]
            return qa, paragraphs

    # fallback: no explicit Q/A found
    return None, lines[2:]


def parse_callstreet_block(text):
    """
    Split by dotted lines, parse each segment into:
      speaker, profession, qa flag, paragraphs
    """
    segments = re.split(r'\.{10,}', text)
    results = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Operator special case
        if segment.startswith("Operator:"):
            op_text = segment.partition(":")[2].strip()
            paragraphs = [
                p.replace("\n", " ").strip()
                for p in re.split(r'\n\s*\n', op_text)
                if p.strip()
            ]

            results.append({
                "speaker": "Operator",
                "profession": "Operator",
                "qa": None,
                "paragraphs": paragraphs
            })
            continue

        lines = [l for l in segment.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        speaker = lines[0].strip()
        profession = lines[1].strip()

        qa, paragraphs = extract_qa_flag_and_paragraphs(lines)

        paragraphs = [
            p.replace("\n", " ").strip()
            for p in re.split(r'\n\s*\n', "\n".join(paragraphs))
            if p.strip()
        ]

        results.append({
            "speaker": speaker,
            "profession": profession,
            "qa": qa,
            "paragraphs": paragraphs
        })

    return results



# ===============================================================
# 4. Fallback line-based parser (rare cases)
# ===============================================================

def parse_fallback(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return [{
        "speaker": None,
        "profession": None,
        "qa": None,
        "paragraphs": lines
    }]



# ===============================================================
# 5. Unified entry point for Q&A parsing
# ===============================================================

def parse_qna_section(text):
    fmt = detect_format(text)

    if fmt == "callstreet":
        return parse_callstreet_block(text)

    if fmt == "inline_qna":
        return parse_inline_qna(text)

    return parse_fallback(text)
