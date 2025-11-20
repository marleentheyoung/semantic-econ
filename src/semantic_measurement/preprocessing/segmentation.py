import re
import pandas as pd

def detect_format(text):
    if re.search(r'\.{10,}', text):
        return "callstreet"
    if re.search(r'<[QA]\s*[–-]\s*[^>]+>', text):
        return "inline_qna"
    return "unknown"


def split_and_extract_speakers(text, is_qna_section=False):
    """Splits text into speaker segments, extracts speaker names + professions, and optionally adds Q/A type.
    Each segment will contain a list of paragraphs instead of a single text block.
    """
    if pd.isna(text) or text.strip() == "":
        return []

    # Split on FactSet-style dotted separator
    segments = re.split(r'\.{10,}', text)

    extracted_segments = []

    for segment in segments:

        segment = segment.strip()
        if not segment:
            continue

        # Special handling for the Operator case
        if segment.startswith("Operator:"):
            speaker = "Operator"
            profession = "Operator"

            # Everything after "Operator:" is the text
            text_after_colon = segment.partition(":")[2].strip()
            
            paragraphs = [p.replace('\n', ' ').strip() for p in text_after_colon.split("\n \n") if p.strip()]
        elif segment.startswith("[Abrupt Start]"):
            continue
        else:
            # Normal Case: Speaker on first line, profession on second line
            lines = segment.split("\n")

            speaker = lines[0]

            try:
                profession = lines[1]
            except:
                continue

            # Everything after "Operator:" is the text
            text_after_colon = '\n'.join(lines[2:])

            paragraphs = [p.replace('\n', ' ').strip() for p in text_after_colon.split("\n \n") if p.strip()]

        # Extract Q/A type in Q&A Section (first paragraph starts with Q or A)
        qa_type = None

        # Pattern 1: First paragraph begins with literal "Q " or "A "
        if is_qna_section and paragraphs:
            first_para = paragraphs[0].strip()

            # Case 1: "Q " or "A "
            if first_para.startswith("Q "):
                qa_type = "Q"
                paragraphs[0] = first_para[2:].strip()

            elif first_para.startswith("A "):
                qa_type = "A"
                paragraphs[0] = first_para[2:].strip()

            else:
                # Case 2: speaker label contains <Q - ...> or <A - ...>
                sp = first_para.strip().lower()
                if sp.startswith("<q"):
                    qa_type = "Q"
                elif sp.startswith("<a"):
                    qa_type = "A"


        # Store result
        if paragraphs:
            segment_data = {
                'speaker': speaker,
                'profession': profession,
                'paragraphs': paragraphs
            }
            if is_qna_section:
                segment_data['qa_type'] = qa_type

            extracted_segments.append(segment_data)

    return extracted_segments

def extract_qa_flag_and_paragraphs(lines):
    qa = None
    # Look at line 2 (index 1) and line 3 (index 2)
    if len(lines) >= 3:
        # If line 2 is exactly "Q" or "A"
        if lines[2].strip() in ("Q", "A"):
            qa = lines[2].strip()
            paragraphs = lines[3:]
            return qa, paragraphs
        
        # If line 3 starts with "Q " or "A "
        if lines[2].strip().startswith("Q "):
            qa = "Q"
            paragraphs = [lines[2].strip()[2:]] + lines[3:]
            return qa, paragraphs
        if lines[2].strip().startswith("A "):
            qa = "A"
            paragraphs = [lines[2].strip()[2:]] + lines[3:]
            return qa, paragraphs
    
    return None, lines[2:]

def parse_inline_qna(text):
    elements = []
    matches = list(pattern.finditer(text))

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
            "profession": None,   # Inline format provides no profession
            "qa": qa,
            "paragraphs": paragraphs
        })

    return elements

def parse_qna_section(text):
    format_type = detect_format(text)

    if format_type == "callstreet":
        return parse_callstreet_block(text)

    if format_type == "inline_qna":
        return parse_inline_qna(text)

    # fallback to line-based speaker parsing
    return parse_fallback(text)

