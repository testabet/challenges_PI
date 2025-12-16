# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta

class ItemsResponse(BaseModel):
    document_id:str
    title: str
    content_snippet:str
    similarity_score: float
    grounded: bool


class DocumentInput(BaseModel):
    title: str
    content: str

class UploadResponse(BaseModel):
    message: str
    document_id: str

class EmbeddingsResponse(BaseModel):
    message: str

class SearchResponse(BaseModel):
    results: List[ItemsResponse]

class AskRequest(BaseModel):
    session_id: str 
    question: str    

class AskResponse(BaseModel):
    question:str
    answer: str
    context_used: List[str]
    similarity_score: List[float]
    grounded: List[bool]
