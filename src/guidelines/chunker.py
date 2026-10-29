import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from src.common.schemas import ParsedDocument, Chunk
from src.guidelines.cleaning import clean_document_pages


class SemanticChunker:
    def __init__(self,base_chunk_size: int = 512,base_overlap: int = 50,adaptive_overlap: bool = True,table_heavy_threshold: int = 200,
                 separators: Optional[List[str]] = None,):
        
        self.base_chunk_size = base_chunk_size
        self.base_overlap = base_overlap
        self.adaptive_overlap = adaptive_overlap
        self.table_heavy_threshold = table_heavy_threshold
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, parsed_doc: ParsedDocument) -> Tuple[List[Chunk], List[dict]]:
        
        overlap = self._determine_overlap(parsed_doc)

        cleaned_pages = self._get_cleaned_pages(parsed_doc)
        combined_text, page_char_boundaries = self._combine_with_boundaries(cleaned_pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.base_chunk_size,
            chunk_overlap=overlap,
            separators=self.separators,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

        chunk_texts = splitter.split_text(combined_text)
        page_ranges = self._map_chunks_to_pages(combined_text, chunk_texts, page_char_boundaries)

        chunks = []
        warnings_list = []

        for index, chunk_text in enumerate(chunk_texts):
            token_count = self._count_tokens(chunk_text)
            start_page, end_page = page_ranges[index]

            chunk = Chunk(
                id=f"{parsed_doc.document_id}_chunk_{index:04d}",
                document_id=parsed_doc.document_id,
                text=chunk_text,
                token_count=token_count,
                chunk_index=index,
                section=None,
                authority_family=parsed_doc.authority_family,
                tier=parsed_doc.tier,
                year=parsed_doc.year,
                drug_names=parsed_doc.drug_names,
                start_page=start_page,
                end_page=end_page,
            )

            chunks.append(chunk)

            outlier_warning = self._detect_outliers(chunk)
            if outlier_warning:
                warnings_list.append(outlier_warning)

            if self._detect_table_split(chunk_text):
                warnings_list.append({
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "warning_category": "table_split_detected",
                    "message": "Chunk starts with table row but no table header detected (possible mid-table split)",
                    "chunk_token_count": token_count,
                    "chunk_index": index,
                    "severity": "medium",
                })

        return chunks, warnings_list


    def _get_cleaned_pages(self, parsed_doc: ParsedDocument) -> List[str]:

        cleaned = getattr(parsed_doc, "cleaned_page_texts", None)
        if cleaned:
            return cleaned

        warnings.warn(
            f"[{parsed_doc.document_id}] cleaned_page_texts not found on "
            f"ParsedDocument — recomputing cleaning on the fly. Run "
            f"clean_parsed_docs.py first so cleaning only happens in one "
            f"place."
        )
        return clean_document_pages(parsed_doc.page_texts)

    def _combine_with_boundaries(self, pages: List[str]) -> Tuple[str, List[int]]:

        parts = []
        boundaries = []
        offset = 0
        separator = "\n\n"

        for page in pages:
            parts.append(page)
            offset += len(page)
            boundaries.append(offset)
            parts.append(separator)
            offset += len(separator)

        return "".join(parts), boundaries

    def _map_chunks_to_pages(self,combined_text: str,chunk_texts: List[str],
                             page_char_boundaries: List[int],) -> List[Tuple[Optional[int], Optional[int]]]:
        page_ranges = []
        search_from = 0

        for chunk_text in chunk_texts:
            if not chunk_text:
                page_ranges.append((None, None))
                continue

            probe_start = max(0, search_from - 200)
            idx = combined_text.find(chunk_text, probe_start)
            if idx == -1:
                idx = combined_text.find(chunk_text)
            if idx == -1:
                page_ranges.append((None, None))
                continue

            start_char = idx
            end_char = idx + len(chunk_text) - 1

            start_page = self._char_offset_to_page(start_char, page_char_boundaries)
            end_page = self._char_offset_to_page(end_char, page_char_boundaries)

            page_ranges.append((start_page, end_page))
            search_from = start_char + 1

        return page_ranges

    @staticmethod
    def _char_offset_to_page(offset: int, boundaries: List[int]) -> Optional[int]:
        if not boundaries:
            return None
        for page_idx, boundary_end in enumerate(boundaries):
            if offset < boundary_end:
                return page_idx
        return len(boundaries) - 1

    def _determine_overlap(self, parsed_doc: ParsedDocument) -> int:
        if not self.adaptive_overlap:
            return self.base_overlap

        if parsed_doc.estimated_table_count > self.table_heavy_threshold:
            return 100
        return self.base_overlap

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _detect_table_split(self, chunk_text: str) -> bool:
        lines = chunk_text.strip().split("\n")
        if not lines or not lines[0].startswith("|"):
            return False
        has_separator = any("| ---" in line or "|---" in line for line in lines[:5])
        return not has_separator

    def _detect_outliers(self, chunk: Chunk) -> Optional[dict]:
        token_count = chunk.token_count

        if token_count < 50:
            severity = self._assign_severity("outlier_too_small", token_count)
            return {
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "warning_category": "outlier_too_small",
                "message": f"Chunk contains only {token_count} tokens (threshold: 50)",
                "chunk_token_count": token_count,
                "chunk_index": chunk.chunk_index,
                "severity": severity,
            }

        if token_count > 800:
            severity = self._assign_severity("outlier_too_large", token_count)
            return {
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "warning_category": "outlier_too_large",
                "message": f"Chunk contains {token_count} tokens (threshold: 800)",
                "chunk_token_count": token_count,
                "chunk_index": chunk.chunk_index,
                "severity": severity,
            }

        return None

    def _assign_severity(self, warning_category: str, token_count: int) -> str:
        if warning_category == "outlier_too_small":
            if token_count >= 40:
                return "low"
            elif token_count >= 20:
                return "medium"
            else:
                return "high"

        elif warning_category == "outlier_too_large":
            if token_count <= 900:
                return "low"
            elif token_count <= 1200:
                return "medium"
            else:
                return "high"

        elif warning_category == "table_split_detected":
            return "medium"

        else:
            return "medium"