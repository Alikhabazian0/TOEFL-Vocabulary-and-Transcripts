import pandas as pd

df = pd.read_csv(r"C:\\Users\\user\\OneDrive\\Desktop\\toefl_vocab_project\\output\\bbc_6min_vocabulary.csv",
encoding = "utf-8-sig")

cleaned = (
    df.groupby("episode_date", group_keys=False)
    .head(3)
    .copy()
)

others = (
    df.drop(cleaned.index)
    .copy()
)

cleaned = cleaned[[
    "episode_date",
    "vocabulary",
    "definition"
]]

cleaned.to_csv("cleaned.csv", index=False, encoding="utf-8-sig")
others.to_csv("others.csv", index=False, encoding="utf-8-sig")

print(f"cleaned.csv: {len(cleaned)} rows")
print(f"others.csv: {len(others)} rows")