from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question, in Arabic or English.")
    chat_history: List[ChatMessage] = Field(default_factory=list)

class SourceItem(BaseModel):
    model_config = ConfigDict(extra="allow")  
    authority_family: Optional[str] = None   
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    document_link: Optional[str] = None

    category: Optional[str] = None
    score: Optional[float] = None

    source_type: Optional[str] = None        
    authority: Optional[str] = None         

    document_id: Optional[str] = None
    chunk_id: Optional[str] = None


class RoutingInfo(BaseModel):
    source: Literal["arabic", "guidelines", "both"]
    language: Literal["ar", "en"]


class ChatResponse(BaseModel):
    answer: str
    chat_history: List[ChatMessage]
    sources: List[SourceItem]
    routing: RoutingInfo
    trust_score: Optional[float] = None
    trust_label: Optional[Literal["high", "medium", "low"]] = None  # FIXED: was missing default


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    detail: Optional[str] = None


class SystemInfoResponse(BaseModel):

    project_name: str = "Medical RAG Assistant"
    supports_languages: List[str] = ["ar", "en"]
    guidelines_sources: List[str] = ["NICE", "FDA"]
    embedding_model: str = "intfloat/multilingual-e5-base"
    vector_store: str = "ChromaDB"
    sparse_retrieval: str = "BM25 (hybrid with dense retrieval, min-max normalized weighted fusion)"
    guidelines_llm: str = "Groq — qwen/qwen3.8-27b"
    arabic_llm: str = "Groq — openai/gpt-oss-120b"
    parsing: str = "LlamaParse (guideline PDFs)"

class XrayResponse(BaseModel):
    label: str                    # "NORMAL" | "PNEUMONIA"
    pneumonia_probability: float
    threshold: float
    disclaimer: str