import json
from pathlib import Path
from dataclasses import asdict

from src.common.schemas import ParsedDocument
from src.guidelines.chunker import SemanticChunker


def build_chunks(
    input_path: str = "data/processed/guidelines/parsed_docs_cleaned.json",
    chunks_output_path: str = "data/processed/chunks.json",
    warnings_output_path: str = "data/processed/chunking_warnings.json",
):
    with open(input_path, "r", encoding="utf-8") as f:
        docs_data = json.load(f)

    chunker = SemanticChunker()

    all_chunks = []
    all_warnings = []

    for doc_data in docs_data:
        parsed_doc = ParsedDocument(**doc_data)
        chunks, warnings = chunker.chunk_document(parsed_doc)

        print(
            f"[OK] {parsed_doc.document_id}: "
            f"{len(chunks)} chunks, {len(warnings)} warning(s)"
        )

        all_chunks.extend(chunks)
        all_warnings.extend(warnings)

    # Serialize chunks (pydantic model -> dict; fall back to dataclasses.asdict
    # if you're using plain dataclasses instead of pydantic for Chunk)
    chunks_serializable = [
        c.model_dump() if hasattr(c, "model_dump") else asdict(c)
        for c in all_chunks
    ]

    Path(chunks_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_serializable, f, ensure_ascii=False, indent=2)

    with open(warnings_output_path, "w", encoding="utf-8") as f:
        json.dump(all_warnings, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_chunks)} chunks -> {chunks_output_path}")
    print(f"Saved {len(all_warnings)} warnings -> {warnings_output_path}")

    # Quick warning summary
    if all_warnings:
        by_category = {}
        for w in all_warnings:
            cat = w["warning_category"]
            by_category[cat] = by_category.get(cat, 0) + 1
        print("\nWarning summary:")
        for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")


if __name__ == "__main__":
    build_chunks()