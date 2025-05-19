from pydantic import BaseModel
from typing import List, Optional

class DocumentUploadResponse(BaseModel):
    filename: str
    message: str
    total_chunks: int

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

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