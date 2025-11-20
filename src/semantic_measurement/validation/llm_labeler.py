"""
LLM-based relevance labeling for threshold calibration.

Labels each candidate paragraph as RELEVANT (1) or NOT RELEVANT (0)
based on the topic and pattern.
"""

from typing import List, Dict, Any
import os

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root if it exists

import anthropic
from semantic_measurement.config.global_calibration import LLM_MODEL


def llm_label_candidates(topic, candidates, model=None, verbose=False):
    model = model or LLM_MODEL

    client = anthropic.Anthropic()
    labeled = []

    example_printed = False

    for c in candidates:
        text = c["text"][:2000]

        prompt = f"""
You are a binary classifier of semantic relevance.

Topic: "{topic}"

Task:
Determine whether this paragraph is relevant to the topic described above.

Output:
Reply ONLY with 'yes' (relevant) or 'no' (not relevant).

Paragraph:
\"\"\"{text}\"\"\"
"""

        # 🔥 Print example prompt once
        if verbose and not example_printed:
            print("\n--- Example LLM Prompt Sent to Claude ---")
            print(prompt)
            print("--- End of Prompt ---\n")
            example_printed = True

        response = client.messages.create(
            model=model,
            max_tokens=5,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.content[0].text.strip().lower()
        is_relevant = 1 if answer == "yes" else 0

        labeled.append({
            **c,
            "label": is_relevant,
        })

    return labeled
