import pandas as pd

first = pd.read_csv(
    "cleaned_repaired.csv",
    encoding="utf-8-sig",
)

second = pd.read_csv(
    "failed_repaired.csv",
    encoding="utf-8-sig",
)

final = pd.concat(
    [first, second],
    ignore_index=True,
)

final = final.drop_duplicates(
    subset=["episode_date", "vocabulary", "definition"]
)

final.to_csv(
    "bbc_vocabulary_final.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"Final vocabulary pairs: {len(final)}")