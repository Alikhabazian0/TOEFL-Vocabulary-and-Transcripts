# TOEFL Vocabulary & Transcripts

Python tools for collecting and organizing vocabulary, expressions, and transcripts from **BBC Learning English’s 6 Minute English**, supporting TOEFL preparation and everyday English learning.

## Features

- Extract vocabulary with definitions and episode metadata.
- Collect full transcripts and speaker-labelled dialogue.
- Clean, repair, merge, and deduplicate vocabulary records.
- Export collected material to CSV and JSON.

## Quick Start

Install the dependencies:

```bash
pip install requests beautifulsoup4 pandas tqdm
```

Run the collectors:

```bash
python collect_vocabulary.py
python collect_transcripts.py
```

Results are saved in the `output/` folder. Adjust `START_DATE` and `END_DATE` in each collector to select your date range.

## Source

Learning material comes from [BBC Learning English — 6 Minute English](https://www.bbc.co.uk/learningenglish/english/features/6-minute-english).

Created by **Ali Khabazian**.
