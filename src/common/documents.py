# src/common/documents.py
"""
Builds a local link to the original source PDF for a document, using
data already present on every chunk/source (document_id + authority_family).
No changes needed to parser.py, chunker.py, or schemas.py — this is a
pure lookup/formatting helper used at the point sources are returned.
"""

from pathlib import Path

# Matches the existing data/raw/<folder>/ layout — see parser.py's
# extract_metadata_from_path, which already derives authority_family
# from filenames following this same convention.
AUTHORITY_TO_FOLDER = {
    "NICE": "nice",
    "FDA": "fda",
    "WHO": "who",
}


def build_document_link(document_id: str, authority_family: str) -> str:
    """
    Returns a deploy-relative link to the original PDF, e.g.
    "/documents/nice/nice_metformin_guideline_diabetes_2021.pdf"

    The API layer is expected to serve data/raw/<folder>/ as static
    files under /documents/<folder>/ (see integration note below).
    """
    folder = AUTHORITY_TO_FOLDER.get(authority_family, "unknown")
    return f"/documents/{folder}/{document_id}.pdf"


def build_document_path_on_disk(document_id: str, authority_family: str) -> Path:
    """
    Returns the actual local filesystem path, useful for the API layer
    when setting up static file serving or checking the file exists.
    """
    folder = AUTHORITY_TO_FOLDER.get(authority_family, "unknown")
    return Path("data/raw") / folder / f"{document_id}.pdf"