from typing import List, Optional

from pydantic import BaseModel, Field

from backend.config import MAX_QUESTION_CHARS, MAX_TOP_K


class DocumentUploadResponse(BaseModel):
    filename: str
    message: str
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(3, ge=1, le=MAX_TOP_K)


class SourceChunk(BaseModel):
    chunk_id: str
    original_content: str
    source_document_name: Optional[str] = None


class SimilarityScore(BaseModel):
    source_chunk_id: str
    source_document_name: Optional[str] = None
    similarity_with_rag_answer: float
    similarity_with_llm_only_answer: float


class QueryResponse(BaseModel):
    rag_answer: str
    rag_sources: List[SourceChunk]
    llm_only_answer: str
    similarity_scores: Optional[List[SimilarityScore]] = None


class ErrorResponse(BaseModel):
    detail: str
