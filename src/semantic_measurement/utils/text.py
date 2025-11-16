# src/semantic_measurement/utils/text.py

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def count_sentences(text: str) -> int:
    """
    Count the number of valid sentences in a paragraph.
    - Uses NLTK's sentence tokenizer (consistent with older pipeline)
    - Filters out trivial/very short fragments

    Returns:
        int: the number of valid sentences
    """
    if not text:
        return 0

    text = str(text).strip()
    if not text:
        return 0

    sentences = sent_tokenize(text)

    # Filter out very short garbage artifacts (< 10 chars)
    valid = [s.strip() for s in sentences if len(s.strip()) > 10]

    return len(valid)
