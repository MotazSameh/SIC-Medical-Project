from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    authority_family: str  
    tier: int              
    year: int
    drug_names: List[str]
    raw_text: str
    parsed_markdown: str
    token_count: int
    page_count: int
    estimated_table_count: int      
    parsing_method: str             
    parse_duration_seconds: float   
    parse_errors: List[str]
    parse_timestamp: str
    page_texts: List[str] = field(default_factory=list)          # موجودة من قبل
    cleaned_page_texts: Optional[List[str]] = None 
    page_texts: List[str] = field(default_factory=list)   


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    token_count: int
    chunk_index: int
    section: Optional[str]
    authority_family: str
    tier: int
    year: Optional[int]
    drug_names: List[str]
    page_number: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None

@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    retriever_type: str  # e.g. "vector", "bm25", "hybrid"
    authority_family: str
    tier: int
    year: Optional[int]
    drug_names: List[str]
    page_number: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None


@dataclass
class GeneratedAnswer:
    question_id: str
    query: str
    answer_text: str
    cited_chunk_ids: List[str]
    is_refusal: bool
    authorities_used: List[str]
    total_token_count: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None


@dataclass
class EvaluationResult:
    question_id: str
    system_variant: str  # e.g. "vector", "bm25", "hybrid"
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    raw_answer: str
    raw_context_ids: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class FailureCase:
    question_id: str
    failure_category: str  # e.g. "retrieval_failure", "hallucination", etc.
    description: str
    diagnosis: str
    proposed_fix: str
    evaluation_result_id: Optional[str] = None
