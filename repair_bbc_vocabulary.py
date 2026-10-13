from __future__ import annotations

import json
import re
import time
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


INPUT_FILE = "cleaned.csv"
OUTPUT_FILE = "cleaned_repaired.csv"
FAILED_FILE = "repair_failed.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

REQUEST_TIMEOUT = 180
MAX_RETRIES = 3


class VocabularyItem(BaseModel):
    vocabulary: str = Field(
        description="The exact vocabulary word, phrase, expression, or idiom."
    )
    definition: str = Field(
        description="The explicit definition given in the source text."
    )


class VocabularyResult(BaseModel):
    items: list[VocabularyItem]


OUTPUT_SCHEMA = VocabularyResult.model_json_schema()


def clean_whitespace(text: Any) -> str:
    """Normalize spaces while safely handling missing values."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_transcript(text: str) -> str:
    """
    Remove transcript content accidentally included after vocabulary.

    Everything after one of these markers is discarded.
    """
    transcript_patterns = [
        r"\bTRANSCRIPT\b",
        r"\bTranscript\b",
        r"\bNote:\s*This is not a word[- ]for[- ]word transcript\b",
        r"\bThis is not a word[- ]for[- ]word transcript\b",
    ]

    cut_positions = []

    for pattern in transcript_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            cut_positions.append(match.start())

    if cut_positions:
        text = text[: min(cut_positions)]

    return clean_whitespace(text)


def build_episode_text(group: pd.DataFrame) -> str:
    """
    Combine vocabulary and definition columns for one episode.

    This handles rows where:
    - multiple vocabulary pairs were merged into one cell;
    - one pair was shifted across two columns;
    - transcript text was accidentally included.
    """
    pieces = []

    for _, row in group.iterrows():
        vocabulary = clean_whitespace(row.get("vocabulary", ""))
        definition = clean_whitespace(row.get("definition", ""))

        if vocabulary:
            pieces.append(vocabulary)

        if definition:
            pieces.append(definition)

    combined = " ".join(pieces)
    combined = remove_transcript(combined)

    return combined


def create_prompt(text: str) -> str:
    return f"""
You are cleaning a BBC Learning English vocabulary section.

Extract every explicitly provided vocabulary-definition pair from the source
text.

Rules:

1. A vocabulary item may be:
   - one word;
   - a phrasal verb;
   - an idiom;
   - an informal expression;
   - several equivalent expressions written together with slashes.

2. Preserve the vocabulary wording as closely as possible.

3. Preserve the definition as closely as possible.

4. Do not extract:
   - speaker names such as Neil, Pippa, Beth, Georgie, Rob or Sam;
   - transcript dialogue;
   - headings such as Vocabulary, Transcript or Note;
   - introductions;
   - questions;
   - examples that are not definitions;
   - invented meanings.

5. Some source text contains several pairs merged together without line breaks.
Separate all meaningful pairs.

6. Return an empty items list when no reliable vocabulary-definition pair is
present.

Example source:
to shoot off to leave somewhere in a hurry fire away an expression giving
permission for someone to ask a question or speak

Expected pairs:
- vocabulary: to shoot off
  definition: to leave somewhere in a hurry
- vocabulary: fire away
  definition: an expression giving permission for someone to ask a question
  or speak

Source text:
{text}
""".strip()


def call_ollama(text: str) -> VocabularyResult:
    """Call local Ollama with a required JSON schema."""
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract explicit vocabulary-definition pairs. "
                    "Never invent information."
                ),
            },
            {
                "role": "user",
                "content": create_prompt(text),
            },
        ],
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data["message"]["content"]
            parsed_json = json.loads(content)

            return VocabularyResult.model_validate(parsed_json)

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


def normalize_vocabulary(text: str) -> str:
    """Normalize vocabulary for duplicate detection."""
    text = clean_whitespace(text).lower()
    text = text.strip(" .,:;!?\"'“”‘’")

    return text


def is_valid_pair(item: VocabularyItem) -> bool:
    """Reject obvious model or source errors."""
    vocabulary = clean_whitespace(item.vocabulary)
    definition = clean_whitespace(item.definition)

    invalid_vocabulary = {
        "transcript",
        "vocabulary",
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

    if vocabulary.lower() in invalid_vocabulary:
        return False

    if len(vocabulary) > 150:
        return False

    if len(definition) < 3:
        return False

    if "this is not a word-for-word transcript" in definition.lower():
        return False

    return True


def main() -> None:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_columns = {
        "episode_date",
        "vocabulary",
        "definition",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    output_records = []
    failed_records = []

    grouped = df.groupby(
        "episode_date",
        sort=False,
        dropna=False,
    )

    for episode_date, group in tqdm(
        grouped,
        total=df["episode_date"].nunique(dropna=False),
        desc="Repairing episodes",
    ):
        source_text = build_episode_text(group)

        if not source_text:
            failed_records.append(
                {
                    "episode_date": episode_date,
                    "reason": "No vocabulary text remained after cleaning",
                    "source_text": "",
                }
            )
            continue

        try:
            result = call_ollama(source_text)

            seen = set()
            episode_count = 0

            for item in result.items:
                if not is_valid_pair(item):
                    continue

                vocabulary = clean_whitespace(item.vocabulary)
                definition = clean_whitespace(item.definition)

                duplicate_key = (
                    normalize_vocabulary(vocabulary),
                    definition.lower(),
                )

                if duplicate_key in seen:
                    continue

                seen.add(duplicate_key)
                episode_count += 1

                output_records.append(
                    {
                        "episode_date": episode_date,
                        "position": episode_count,
                        "vocabulary": vocabulary,
                        "definition": definition,
                    }
                )

            if episode_count == 0:
                failed_records.append(
                    {
                        "episode_date": episode_date,
                        "reason": "Model returned no valid pairs",
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

    repaired_df = pd.DataFrame(output_records)

    if not repaired_df.empty:
        repaired_df = repaired_df.sort_values(
            ["episode_date", "position"],
            kind="stable",
        )

    repaired_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(failed_records).to_csv(
        FAILED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished")
    print(f"Vocabulary pairs: {len(repaired_df)}")
    print(f"Failed episodes: {len(failed_records)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Failures: {FAILED_FILE}")


if __name__ == "__main__":
    main()