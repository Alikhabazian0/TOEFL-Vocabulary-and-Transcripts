from __future__ import annotations

from datetime import date

import pandas as pd
from tqdm import tqdm

from bbc_common import (
    OUTPUT_DIR,
    download_episode,
    extract_between_headings,
    generate_thursdays,
    get_episode_title,
    make_soup,
    remove_leading_transcript_note,
    trim_transcript_at_goodbye,
)


START_DATE = date(2017, 2, 23)
END_DATE = date(2026, 7, 30)


def lines_to_dialogue_rows(
    lines: list[str],
) -> list[dict[str, str | int | None]]:
    """
    Convert transcript lines into speaker/dialogue records.

    Typical input:
        Neil
        Hello and welcome...
        Becca
        Today we're talking about...

    Speaker names are detected as short lines without sentence punctuation.
    """
    rows: list[dict[str, str | int | None]] = []

    current_speaker: str | None = None
    dialogue_position = 0

    for line in lines:
        line = line.strip()

        if not line:
            continue

        is_probable_speaker = (
            len(line) <= 50
            and len(line.split()) <= 6
            and not line.endswith((".", "?", "!", ",", ";", ":"))
        )

        special_speaker_labels = {
            "both",
            "insert",
            "narrator",
        }

        if (
            is_probable_speaker
            or line.lower() in special_speaker_labels
        ):
            current_speaker = line
            continue

        dialogue_position += 1

        rows.append(
            {
                "dialogue_position": dialogue_position,
                "speaker": current_speaker,
                "text": line,
            }
        )

    return rows


def main() -> None:
    dialogue_records: list[dict] = []
    episode_records: list[dict] = []
    failed_dates: list[dict] = []

    dates = list(generate_thursdays(START_DATE, END_DATE))

    for episode_date in tqdm(dates, desc="Collecting transcripts"):
        downloaded = download_episode(episode_date)

        if downloaded is None:
            failed_dates.append(
                {
                    "expected_date": episode_date.isoformat(),
                    "reason": "Page not found",
                }
            )
            continue

        url, html = downloaded
        soup = make_soup(html)
        title = get_episode_title(soup)

        # There is normally no relevant heading after Transcript,
        # so extraction stops at the next major heading.
        transcript_lines = extract_between_headings(
            soup=soup,
            start_heading="Transcript",
            end_heading=None,
        )

        transcript_lines = remove_leading_transcript_note(
            transcript_lines
        )

        transcript_lines = trim_transcript_at_goodbye(
            transcript_lines
        )

        if not transcript_lines:
            failed_dates.append(
                {
                    "expected_date": episode_date.isoformat(),
                    "reason": "Transcript not extracted",
                    "url": url,
                }
            )
            continue

        rows = lines_to_dialogue_rows(transcript_lines)

        for row in rows:
            dialogue_records.append(
                {
                    "episode_date": episode_date.isoformat(),
                    "episode_code": episode_date.strftime("%y%m%d"),
                    "episode_title": title,
                    "dialogue_position": row["dialogue_position"],
                    "speaker": row["speaker"],
                    "text": row["text"],
                    "source_url": url,
                }
            )

        full_transcript = "\n".join(transcript_lines)

        episode_records.append(
            {
                "episode_date": episode_date.isoformat(),
                "episode_code": episode_date.strftime("%y%m%d"),
                "episode_title": title,
                "full_transcript": full_transcript,
                "source_url": url,
            }
        )

    # One row for each speaker turn/dialogue line.
    dialogue_df = pd.DataFrame(dialogue_records)

    dialogue_df.to_csv(
        OUTPUT_DIR / "bbc_6min_dialogue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dialogue_df.to_json(
        OUTPUT_DIR / "bbc_6min_dialogue.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # One row for each complete episode.
    episodes_df = pd.DataFrame(episode_records)

    episodes_df.to_csv(
        OUTPUT_DIR / "bbc_6min_full_transcripts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    episodes_df.to_json(
        OUTPUT_DIR / "bbc_6min_full_transcripts.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    pd.DataFrame(failed_dates).to_csv(
        OUTPUT_DIR / "transcript_failed_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished.")
    print(f"Transcripts collected: {len(episodes_df)}")
    print(f"Dialogue rows collected: {len(dialogue_df)}")
    print(f"Failed dates: {len(failed_dates)}")


if __name__ == "__main__":
    main()