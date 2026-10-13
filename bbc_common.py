from __future__ import annotations

import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.bbc.co.uk/learningenglish/english/features"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130 Safari/537.36"
    )
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update(HEADERS)


def clean_text(text: str) -> str:
    """Normalize whitespace and special spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def generate_thursdays(
    start_date: date,
    end_date: date,
) -> Iterator[date]:
    """
    Generate every Thursday between two dates.

    Python weekday:
    Monday = 0
    Thursday = 3
    """
    current = start_date

    days_until_thursday = (3 - current.weekday()) % 7
    current += timedelta(days=days_until_thursday)

    while current <= end_date:
        yield current
        current += timedelta(days=7)


def episode_code(episode_date: date) -> str:
    """Convert 2026-07-30 to 260730."""
    return episode_date.strftime("%y%m%d")


def candidate_urls(episode_date: date) -> list[str]:
    """
    BBC has used at least two URL structures.

    Newer:
    /6-minute-english_2026/ep-260730

    Older:
    /6-minute-english/ep-170223
    """
    code = episode_code(episode_date)
    year = episode_date.year

    return [
        f"{BASE_URL}/6-minute-english_{year}/ep-{code}",
        f"{BASE_URL}/6-minute-english/ep-{code}",
    ]


def download_episode(
    episode_date: date,
    delay: float = 0.7,
) -> tuple[str, str] | None:
    """
    Try both known URL structures.

    Returns:
        (working_url, html)

    Returns None when no page is found.
    """
    for url in candidate_urls(episode_date):
        try:
            response = session.get(url, timeout=30)

            if response.status_code == 200:
                html = response.text

                # Prevent false positives such as generic BBC error pages.
                if (
                    "6 Minute English" in html
                    and "Transcript" in html
                    and "Vocabulary" in html
                ):
                    time.sleep(delay)
                    return url, html

            time.sleep(delay)

        except requests.RequestException as error:
            print(f"Request error for {url}: {error}")

    return None


def make_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        ["script", "style", "noscript", "svg", "nav", "footer"]
    ):
        element.decompose()

    return soup


def heading_matches(tag: Tag, target: str) -> bool:
    """Check whether a heading matches a requested section."""
    if tag.name not in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return False

    heading_text = clean_text(tag.get_text(" ", strip=True))
    heading_text = heading_text.rstrip(":").lower()

    return heading_text == target.lower().rstrip(":")


def find_heading(
    soup: BeautifulSoup,
    target: str,
) -> Tag | None:
    for heading in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        if heading_matches(heading, target):
            return heading

    return None


def extract_between_headings(
    soup: BeautifulSoup,
    start_heading: str,
    end_heading: str | None = None,
) -> list[str]:
    """
    Extract visible text after one heading and before another heading.

    Example:
        Vocabulary -> Transcript
    """
    start = find_heading(soup, start_heading)

    if start is None:
        return []

    lines: list[str] = []

    for element in start.find_all_next():
        if element == start:
            continue

        if isinstance(element, Tag):
            if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                heading_text = clean_text(
                    element.get_text(" ", strip=True)
                ).rstrip(":")

                if end_heading is None:
                    break

                if heading_text.lower() == end_heading.lower().rstrip(":"):
                    break

            # Only collect common content containers.
            if element.name in {"p", "li"}:
                text = clean_text(element.get_text(" ", strip=True))

                if text and text not in lines:
                    lines.append(text)

    return lines


def get_episode_title(soup: BeautifulSoup) -> str | None:
    """
    Extract the programme title.

    We avoid returning the generic '6 Minute English' heading.
    """
    headings = soup.find_all(["h1", "h2", "h3"])

    for heading in headings:
        text = clean_text(heading.get_text(" ", strip=True))

        if (
            text
            and text.lower() != "6 minute english"
            and not text.lower().startswith("episode ")
            and text.lower() not in {
                "vocabulary",
                "transcript",
                "this week's question",
            }
        ):
            return text

    return None


def remove_leading_transcript_note(lines: list[str]) -> list[str]:
    """Remove BBC's transcript disclaimer."""
    note_patterns = [
        "this is not a word-for-word transcript",
        "this is not a word for word transcript",
        "note: this is not a word-for-word transcript",
        "note: this is not a word for word transcript",
    ]

    cleaned: list[str] = []

    for line in lines:
        normalized = clean_text(line).lower().rstrip(".")

        if any(pattern in normalized for pattern in note_patterns):
            continue

        cleaned.append(clean_text(line))

    return cleaned


def trim_transcript_at_goodbye(lines: list[str]) -> list[str]:
    """
    Retain the transcript through its final goodbye.

    The function uses the last goodbye because 'bye' might be mentioned
    earlier as part of a conversation.
    """
    goodbye_pattern = re.compile(
        r"^(?:both\s+)?(?:goodbye|bye|bye-bye|bye bye)[!. ]*$",
        flags=re.IGNORECASE,
    )

    last_goodbye_index: int | None = None

    for index, line in enumerate(lines):
        if goodbye_pattern.fullmatch(clean_text(line)):
            last_goodbye_index = index

    if last_goodbye_index is not None:
        return lines[: last_goodbye_index + 1]

    return lines