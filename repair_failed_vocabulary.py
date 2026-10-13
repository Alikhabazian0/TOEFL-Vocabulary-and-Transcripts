from __future__ import annotations

import json
import re
import time
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


INPUT_FILE = "repair_failed.csv"
OUTPUT_FILE = "failed_repaired.csv"
STILL_FAILED_FILE = "still_failed.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 180


class VocabularyPair(BaseModel):
    vocabulary: str = Field(description="Vocabulary word or expression")
    definition: str = Field(description="Meaning or definition")


class ExtractionResult(BaseModel):
    items: list[VocabularyPair]


JSON_SCHEMA = ExtractionResult.model_json_schema()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_unwanted_text(text: str) -> str:
    """Remove transcript material if it remains in source_text."""
    markers = [
        r"\bTRANSCRIPT\b",
        r"\bNote:\s*This is not a word[- ]for[- ]word transcript\b",
        r"\bThis is not a word[- ]for[- ]word transcript\b",
    ]

    positions = []

    for marker in markers:
        match = re.search(marker, text, flags=re.IGNORECASE)

        if match:
            positions.append(match.start())

    if positions:
        text = text[: min(positions)]

    return clean_text(text)


def build_prompt(source_text: str) -> str:
    return f"""
Extract all vocabulary-definition pairs from the text.

The text contains consecutive entries in this form:

VOCABULARY TERM + DEFINITION + NEXT VOCABULARY TERM + NEXT DEFINITION

There may be no separators or line breaks.

Important rules:
- Extract every vocabulary word, phrasal verb, expression, or idiom.
- Separate each vocabulary item from its definition.
- Do not rewrite or invent meanings.
- Do not combine multiple vocabulary items into one item.
- Keep expressions such as "pay back / pay off / clear debt" together when
  the source treats them as equivalent expressions.
- Ignore headings, speaker names, transcript dialogue, and unrelated text.
- Return as many reliable pairs as the source contains.
- Normally there are approximately 3 to 10 pairs.

Example 1:

Input:
to shoot off to leave somewhere in a hurry fire away an expression giving
permission for someone to ask a question or speak restrain prevent someone
from moving

Output:
1. vocabulary: to shoot off
   definition: to leave somewhere in a hurry

2. vocabulary: fire away
   definition: an expression giving permission for someone to ask a question
   or speak

3. vocabulary: restrain
   definition: prevent someone from moving

Example 2:

Input:
debt money a person has borrowed and owes to someone make ends meet have just
enough money to pay for basic living expenses financial strain emotional
stress caused by a lack of money

Output:
1. vocabulary: debt
   definition: money a person has borrowed and owes to someone

2. vocabulary: make ends meet
   definition: have just enough money to pay for basic living expenses

3. vocabulary: financial strain
   definition: emotional stress caused by a lack of money

Now extract the entries from this source:

{source_text}
""".strip()


def call_ollama(source_text: str) -> ExtractionResult:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "format": JSON_SCHEMA,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1500,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise vocabulary extraction system. "
                    "Return only vocabulary-definition pairs supported "
                    "by the source."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(source_text),
            },
        ],
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            response_data = response.json()
            model_content = response_data["message"]["content"]

            parsed = json.loads(model_content)

            return ExtractionResult.model_validate(parsed)

        except (
            requests.RequestException,
            KeyError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            last_error = error

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Ollama failed after {MAX_RETRIES} attempts: {last_error}"
    )


def normalize_for_duplicate_check(text: str) -> str:
    text = clean_text(text).lower()
    text = text.strip(" .,:;!?\"'“”‘’")

    return text


def valid_pair(pair: VocabularyPair) -> bool:
    vocabulary = clean_text(pair.vocabulary)
    definition = clean_text(pair.definition)

    invalid_terms = {
        "vocabulary",
        "transcript",
        "note",
        "neil",
        "pippa",
        "beth",
        "becca",
        "georgie",
        "rob",
        "sam",
    }

    if not vocabulary or not definition:
        return False

    if vocabulary.lower() in invalid_terms:
        return False

    # Vocabulary expressions can occasionally be fairly long.
    if len(vocabulary) > 200:
        return False

    if len(definition) < 2:
        return False

    if vocabulary.lower() == definition.lower():
        return False

    return True


def main() -> None:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_columns = {
        "episode_date",
        "source_text",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    extracted_records = []
    failed_records = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Repairing failed episodes",
    ):
        episode_date = clean_text(row["episode_date"])
        source_text = remove_unwanted_text(
            clean_text(row["source_text"])
        )

        if not source_text:
            failed_records.append(
                {
                    "episode_date": episode_date,
                    "reason": "Empty source text",
                    "source_text": source_text,
                }
            )
            continue

        try:
            result = call_ollama(source_text)

            seen = set()
            position = 0

            for pair in result.items:
                if not valid_pair(pair):
                    continue

                vocabulary = clean_text(pair.vocabulary)
                definition = clean_text(pair.definition)

                duplicate_key = (
                    normalize_for_duplicate_check(vocabulary),
                    normalize_for_duplicate_check(definition),
                )

                if duplicate_key in seen:
                    continue

                seen.add(duplicate_key)
                position += 1

                extracted_records.append(
                    {
                        "episode_date": episode_date,
                        "position": position,
                        "vocabulary": vocabulary,
                        "definition": definition,
                    }
                )

            if position == 0:
                failed_records.append(
                    {
                        "episode_date": episode_date,
                        "reason": "Second pass returned no valid pairs",
                        "source_text": source_text,
                    }
                )

        except Exception as error:
            failed_records.append(
                {
                    "episode_date": episode_date,
                    "reason": str(error),
                    "source_text": source_text,
                }
            )

    extracted_df = pd.DataFrame(extracted_records)
    failed_df = pd.DataFrame(failed_records)

    extracted_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failed_df.to_csv(
        STILL_FAILED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished")
    print(f"Extracted vocabulary pairs: {len(extracted_df)}")
    print(f"Successfully repaired episodes: "
          f"{extracted_df['episode_date'].nunique() if not extracted_df.empty else 0}")
    print(f"Still-failed episodes: {len(failed_df)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Remaining failures: {STILL_FAILED_FILE}")


if __name__ == "__main__":
    main()