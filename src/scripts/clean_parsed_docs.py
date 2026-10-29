import json
from pathlib import Path

from src.guidelines.cleaning import clean_document_pages


def clean_parsed_docs(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    for doc in docs:
        page_texts = doc.get("page_texts", [])
        if not page_texts:
            print(f"[SKIP] {doc.get('document_id')}: no page_texts found")
            doc["cleaned_page_texts"] = []
            continue

        cleaned = clean_document_pages(page_texts)
        doc["cleaned_page_texts"] = cleaned

        # quick before/after size check per doc
        before_chars = sum(len(p) for p in page_texts)
        after_chars = sum(len(p) for p in cleaned)
        reduction_pct = 100 * (1 - after_chars / before_chars) if before_chars else 0
        print(
            f"[OK] {doc.get('document_id')}: "
            f"{before_chars} -> {after_chars} chars "
            f"({reduction_pct:.1f}% removed)"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cleaned docs -> {output_path}")


if __name__ == "__main__":
    clean_parsed_docs(
        input_path="data/processed/guidelines/parsed_docs_debug.json",
        output_path="data/processed/guidelines/parsed_docs_cleaned.json",
    )