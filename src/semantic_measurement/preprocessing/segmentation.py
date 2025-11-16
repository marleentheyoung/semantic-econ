import re
import pandas as pd

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
        if is_qna_section and paragraphs:
            first_para = paragraphs[0].strip()
            if first_para.startswith('Q '):
                qa_type = 'Q'
                paragraphs[0] = first_para[2:].strip()
            elif first_para.startswith('A '):
                qa_type = 'A'
                paragraphs[0] = first_para[2:].strip()

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