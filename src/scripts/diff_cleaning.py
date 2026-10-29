import argparse
import difflib
import json
import sys
from pathlib import Path


def load_doc(cleaned_path: str, document_id: str):
    with open(cleaned_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    for doc in docs:
        if doc.get("document_id") == document_id:
            return doc

    print(f"[ERROR] document_id '{document_id}' not found in {cleaned_path}")
    sys.exit(1)


def diff_pages(doc: dict, page_indices=None):
    raw_pages = doc.get("page_texts", [])
    cleaned_pages = doc.get("cleaned_page_texts", [])

    if len(raw_pages) != len(cleaned_pages):
        print(
            f"[WARN] page count mismatch: raw={len(raw_pages)} "
            f"cleaned={len(cleaned_pages)}"
        )

    n_pages = min(len(raw_pages), len(cleaned_pages))
    indices = page_indices if page_indices else range(n_pages)

    total_removed_lines = 0

    for i in indices:
        if i >= n_pages:
            print(f"[SKIP] page index {i} out of range (doc has {n_pages} pages)")
            continue

        raw_lines = raw_pages[i].split("\n")
        cleaned_lines = cleaned_pages[i].split("\n")

        sm = difflib.SequenceMatcher(a=raw_lines, b=cleaned_lines)
        removed_this_page = []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "delete" or tag == "replace":
                removed_this_page.extend(raw_lines[i1:i2])

        if removed_this_page:
            print(f"\n{'=' * 70}")
            print(f"PAGE {i} — {len(removed_this_page)} line(s) removed")
            print(f"{'=' * 70}")
            for line in removed_this_page:
                marker = "(blank)" if not line.strip() else ""
                print(f"  - {line!r} {marker}")
            total_removed_lines += len(removed_this_page)
        else:
            print(f"\nPAGE {i}: no changes")

    print(f"\n{'#' * 70}")
    print(f"TOTAL removed lines across shown pages: {total_removed_lines}")
    print(f"{'#' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Diff page_texts vs cleaned_page_texts")
    parser.add_argument("document_id", help="document_id to inspect")
    parser.add_argument(
        "--file",
        default="data/processed/guidelines/parsed_docs_cleaned.json",
        help="Path to the cleaned docs JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="*",
        default=None,
        help="Specific 0-indexed page numbers to inspect (default: all pages)",
    )
    args = parser.parse_args()

    doc = load_doc(args.file, args.document_id)
    print(f"Document: {doc.get('document_id')}")
    print(f"Total pages: {doc.get('page_count')}")
    diff_pages(doc, args.pages)


if __name__ == "__main__":
    main()