from pathlib import Path
import re
import unicodedata

import pandas as pd
from sqlalchemy import true


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/arabic/data.csv")
OUTPUT_DIR = Path("data/arabic")
OUTPUT_FILE = OUTPUT_DIR / "data_cleaned.csv"

REQUIRED_COLUMNS = {"category","question","answer",}


def remove_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_arabic_diacritics(text: str) -> str:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    return text


def remove_tatweel(text: str) -> str:
    return text.replace("ـ", "")


def normalize_arabic_letters(text: str) -> str:
    translation_table = str.maketrans({
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",

        "ک": "ك",
        "ی": "ي",
        "ئ": "ي",

        "ؤ": "و",

        "ھ": "ه",
        "ہ": "ه",
        "ە": "ه",
    })

    return text.translate(translation_table)


def normalize_line_breaks(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = remove_html(text)
    text = normalize_unicode(text)
    text = normalize_line_breaks(text)
    text = remove_arabic_diacritics(text)
    text = remove_tatweel(text)
    text = normalize_arabic_letters(text)
    text = normalize_whitespace(text)
    return text


# ============================================================
# Validation
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    actual_columns = set(df.columns)
    missing = REQUIRED_COLUMNS - actual_columns

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )


def print_dataset_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("ARABIC DATASET REPORT")
    print("=" * 60)

    print(f"Total rows: {len(df):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")

    for column in ["category", "question", "answer"]:
        missing = df[column].isna().sum()
        print(f"  {column}: {missing:,}")

    print("\nEmpty cleaned values:")

    for column in ["clean_category","clean_question","clean_answer",]:
        empty = (df[column].str.strip() == "").sum()
        print(f"  {column}: {empty:,}")

    print("\nUnique categories:")
    print(f"  {df['clean_category'].nunique():,}")

    print("\nTop categories:")

    category_counts = (df["clean_category"].value_counts().head(15))

    for category, count in category_counts.items():
        print(f"  {category}: {count:,}")

    print("\nQuestion length:")
    print(f"  Mean:   {df['clean_question'].str.len().mean():.1f}")
    print(f"  Median: {df['clean_question'].str.len().median():.1f}")
    print(f"  Max:    {df['clean_question'].str.len().max():,}")

    print("\nAnswer length:")
    print(f"  Mean:   {df['clean_answer'].str.len().mean():.1f}")
    print(f"  Median: {df['clean_answer'].str.len().median():.1f}")
    print(f"  Max:    {df['clean_answer'].str.len().max():,}")

    print("=" * 60)


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("=" * 60)
    print("Arabic Medical Q&A Cleaning")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    print(f"\nReading: {INPUT_FILE}")


    df = pd.read_csv(INPUT_FILE,encoding="utf-8-sig",)

    print(f"Loaded {len(df):,} rows.")

    validate_columns(df)

    print("\nInput schema validated.")


    print("\nCleaning Arabic text...")

    df["clean_category"] = df["category"].apply(clean_text)
    df["clean_question"] = df["question"].apply(clean_text)
    df["clean_answer"] = df["answer"].apply(clean_text)


    before_duplicates = len(df)

    duplicate_mask = df.duplicated( subset=["clean_category", "clean_question","clean_answer",],keep="first",)

    duplicate_count = duplicate_mask.sum()

    print(f"\nExact duplicate rows detected: {duplicate_count:,}")


    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,)

    df.to_csv( OUTPUT_FILE,index=False,encoding="utf-8-sig",)

    print_dataset_report(df)

    print("\nOutput:")
    print(f"  {OUTPUT_FILE.resolve()}")

    print("\nRows:")
    print(f"  Input:  {before_duplicates:,}")
    print(f"  Output: {len(df):,}")

    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()