# src/common/evidence.py
"""
Normalized evidence representation shared between the Guidelines RAG
(NICE/FDA) and the Arabic Medical Q&A RAG.

Both retrievers keep their own internal result types (RetrievedChunk for
Guidelines, plain dicts for Arabic) — this module only adapts their
existing outputs into one common shape. It does not change how either
retriever works internally.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class NormalizedEvidence:
    """
    Common evidence shape consumed by the router and by generation.

    authority is intentionally separate from source_type:
    - source_type identifies WHERE the evidence came from ("guidelines" / "arabic_qa")
    - authority identifies its official weight, if any ("NICE" / "FDA" / None)
      Arabic Q&A has no formal authority — it must never be presented
      with the same weight as an official guideline (see Rule 8).
    """
    source_type: str          # "guidelines" | "arabic_qa"
    source_id: str            # chunk_id (kept for traceability / citations)
    document_id: str
    chunk_id: str
    text: str
    score: float
    rank: int
    language: str             # "en" | "ar"
    authority: Optional[str]  # "NICE" | "FDA" | None (Arabic Q&A has none)
    metadata: Dict[str, Any] = field(default_factory=dict)


def from_guidelines_chunk(chunk, rank: int, retriever_type: str = "hybrid") -> NormalizedEvidence:
    """
    Adapts a Guidelines RetrievedChunk (from hybrid_retrieve / HybridRetriever)
    into NormalizedEvidence. Reads attributes only — does not modify the
    original chunk object.
    """
    return NormalizedEvidence(
        source_type="guidelines",
        source_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        score=chunk.score,
        rank=rank,
        language="en",
        authority=chunk.authority_family,  # "NICE" | "FDA" | "WHO" | "UNKNOWN"
        metadata={
            "tier": chunk.tier,
            "year": chunk.year,
            "drug_names": chunk.drug_names,
            "start_page": getattr(chunk, "start_page", None),
            "end_page": getattr(chunk, "end_page", None),
            "retriever_type": retriever_type,
        },
    )


def from_arabic_hit(hit: Dict[str, Any]) -> NormalizedEvidence:
    """
    Adapts a single hit dict from dense_retrieve() (Arabic pipeline)
    into NormalizedEvidence. Reads keys only — does not modify the
    original dict.
    """
    meta = hit.get("meta", {})
    return NormalizedEvidence(
        source_type="arabic_qa",
        source_id=str(hit["id"]),
        document_id=meta.get("document_id", str(hit["id"])),
        chunk_id=str(hit["id"]),
        text=hit["text"],
        score=hit["score"],
        rank=hit.get("rank", 0),
        language="ar",
        authority=None,  # Arabic Q&A is never an official authority
        metadata={
            "category": meta.get("category", ""),
            "token_count": meta.get("token_count", 0),
        },
    )


def normalize_guidelines_results(chunks, retriever_type: str = "hybrid") -> List[NormalizedEvidence]:
    """Adapts a full list of Guidelines RetrievedChunk results, preserving order as rank."""
    return [
        from_guidelines_chunk(c, rank=i, retriever_type=retriever_type)
        for i, c in enumerate(chunks)
    ]


def normalize_arabic_results(hits: List[Dict[str, Any]]) -> List[NormalizedEvidence]:
    """Adapts a full list of hit dicts from dense_retrieve()."""
    return [from_arabic_hit(h) for h in hits]