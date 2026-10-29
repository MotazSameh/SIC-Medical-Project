import json
from dataclasses import asdict
from pathlib import Path
from src.common.settings_config import settings
from src.guidelines.parser import DocumentParser

def dump_parsed_docs(raw_dirs: list[str], output_path: str):
    parser = DocumentParser(api_key=settings.llama_cloud_api_key)
    all_docs = []

    for raw_dir in raw_dirs:
        for pdf_path in Path(raw_dir).glob("*.pdf"):
            print(f"Parsing: {pdf_path}")
            try:
                parsed = parser.parse_pdf(str(pdf_path))
                all_docs.append(asdict(parsed))
            except Exception as e:
                print(f"[FAILED] {pdf_path}: {e}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_docs)} parsed docs -> {output_path}")


if __name__ == "__main__":
    dump_parsed_docs(
        raw_dirs=["data/raw/nice", "data/raw/fda"],
        output_path="data/processed/guidelines/parsed_docs_debug.json"
    )