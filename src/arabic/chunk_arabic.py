
from pathlib import Path
import json
import re

import pandas as pd

INPUT_FILE = Path("data/arabic/data_cleaned.csv")

OUTPUT_DIR = Path("data/arabic")

OUTPUT_FILE = OUTPUT_DIR / "arabic_chunks.json"

LANGUAGE = "ar"
SOURCE_TYPE = "arabic_qa"


def estimate_token_count(text: str) -> int:

    if not text or not text.strip():
        return 0
    
    return len(text.split())


def normalize_category(category: str) -> str:

    if pd.isna(category):
        return ""

    return str(category).strip()


def build_chunk_text(question: str,answer: str,) -> str:

    question = str(question).strip()
    answer = str(answer).strip()

    return (f"السؤال:\n"f"{question}\n\n"f"الإجابة:\n"f"{answer}")


def validate_input(df: pd.DataFrame) -> None:

    required_columns = { "clean_category","clean_question","clean_answer",}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError("Missing required columns in cleaned dataset: "f"{sorted(missing)}")


def build_chunks(df: pd.DataFrame) -> list:

    chunks = []

    for index, row in df.iterrows():

        category = normalize_category(row["clean_category"])

        question = str(row["clean_question"]).strip()

        answer = str(row["clean_answer"]).strip()

        if not question or not answer:
            continue

        chunk_id = f"arabic_qa_{index:06d}"

        document_id = chunk_id

        text = build_chunk_text(question=question,answer=answer,)


        chunk = {
            "id": chunk_id,
            "document_id": document_id,
            "text": text,
            "token_count": estimate_token_count(text),
            "chunk_index": 0,
            "section": None,
            "authority_family": "ARABIC_QA",
            "tier": None,
            "year": None,
            "drug_names": [],
            "page_number": None,
            "start_page": None,
            "end_page": None,
            "language": "ar",
            "source_type": "arabic_qa",
            "category": category,
            "question": question,
            "answer": answer,
        }

        chunks.append(chunk)

    return chunks


# ============================================================
# Validation Report
# ============================================================

def print_chunk_report(chunks: list) -> None:

    print("\n" + "=" * 60)
    print("ARABIC CHUNK REPORT")
    print("=" * 60)

    print(f"\nTotal chunks: {len(chunks):,}")

    if not chunks:
        print("\nNo chunks were generated.")
        return

    categories = {}

    for chunk in chunks:

        category = chunk["category"]

        categories[category] = (categories.get(category, 0) + 1)

    print(f"Unique categories: {len(categories):,}")

    print("\nTop categories:")

    sorted_categories = sorted(categories.items(),key=lambda x: x[1],reverse=True,)

    for category, count in sorted_categories[:15]:

        print(f"  {category}: {count:,}")

    token_counts = [chunk["token_count"]for chunk in chunks]

    print("\nEstimated token counts:")

    print(f"  Mean: {sum(token_counts) / len(token_counts):.1f}")

    print(f"  Min:  {min(token_counts):,}")

    print(f"  Max:  {max(token_counts):,}")


    long_chunks = [chunk for chunk in chunks if chunk["token_count"] > 1000]

    print(f"\nChunks > 1000 estimated tokens: "f"{len(long_chunks):,}")

    empty_question = sum(not chunk["question"] for chunk in chunks)

    empty_answer = sum(not chunk["answer"]for chunk in chunks)

    print("\nEmpty fields:")

    print(f"  Empty questions: {empty_question:,}")

    print(f"  Empty answers:   {empty_answer:,}")

    print("=" * 60)


def main():

    print("=" * 60)
    print("Arabic Medical Q&A Chunk Builder")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cleaned dataset not found:\n"f"{INPUT_FILE.resolve()}")

    print(f"\nInput: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE,encoding="utf-8-sig",)

    print(f"Loaded rows: {len(df):,}")

    validate_input(df)

    print("Input schema validated successfully.")

    print("\nBuilding Arabic Q&A chunks...")

    chunks = build_chunks(df)

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,)

    with open(OUTPUT_FILE,"w",encoding="utf-8",) as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2,)

    print_chunk_report(chunks)

    print("\nOutput:")
    print(f"  {OUTPUT_FILE.resolve()}")

    print("\nChunking completed successfully.")

if __name__ == "__main__":
    main()