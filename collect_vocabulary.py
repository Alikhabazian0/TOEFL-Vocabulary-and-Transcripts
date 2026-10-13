from __future__ import annotations

from datetime import date

import pandas as pd
from tqdm import tqdm

from bbc_common import (
    OUTPUT_DIR,
    clean_text,
    download_episode,
    extract_between_headings,
    generate_thursdays,
    get_episode_title,
    make_soup,
)


START_DATE = date(2017, 2, 23)
END_DATE = date(2026, 7, 30)


def parse_vocabulary_lines(lines: list[str]) -> list[dict[str, str]]:
    """
    BBC usually presents vocabulary as alternating elements:

        term
        definition
        term
        definition

    This parser groups every two lines.
    """
    cleaned = [
        clean_text(line)
        for line in lines
        if clean_text(line)
    ]

    entries: list[dict[str, str]] = []

    index = 0

    while index < len(cleaned):
        term = cleaned[index]

        definition = (
            cleaned[index + 1]
            if index + 1 < len(cleaned)
            else ""
        )

        entries.append(
            {
                "vocabulary": term,
                "definition": definition,
            }
        )

        index += 2

    return entries


def main() -> None:
    records: list[dict] = []
    failed_dates: list[dict] = []

    dates = list(generate_thursdays(START_DATE, END_DATE))

    for episode_date in tqdm(dates, desc="Collecting vocabulary"):
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

        vocabulary_lines = extract_between_headings(
            soup=soup,
            start_heading="Vocabulary",
            end_heading="Transcript",
        )

        vocabulary_entries = parse_vocabulary_lines(vocabulary_lines)

        if not vocabulary_entries:
            failed_dates.append(
                {
                    "expected_date": episode_date.isoformat(),
                    "reason": "Vocabulary section not extracted",
                    "url": url,
                }
            )
            continue

        for position, entry in enumerate(
            vocabulary_entries,
            start=1,
        ):
            records.append(
                {
                    "episode_date": episode_date.isoformat(),
                    "episode_code": episode_date.strftime("%y%m%d"),
                    "episode_title": title,
                    "position": position,
                    "vocabulary": entry["vocabulary"],
                    "definition": entry["definition"],
                    "source_url": url,
                }
            )

    vocabulary_df = pd.DataFrame(records)

    vocabulary_df.to_csv(
        OUTPUT_DIR / "bbc_6min_vocabulary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vocabulary_df.to_json(
        OUTPUT_DIR / "bbc_6min_vocabulary.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    pd.DataFrame(failed_dates).to_csv(
        OUTPUT_DIR / "vocabulary_failed_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished.")
    print(f"Episodes/entries collected: {len(vocabulary_df)}")
    print(f"Failed dates: {len(failed_dates)}")


if __name__ == "__main__":
    main()