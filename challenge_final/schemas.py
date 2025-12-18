from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ItemsEvidence(BaseModel):
    text: str
    score: float = Field(None, ge=0, le=1, description="Similaridad (valoces cercanos a 1 representan mayor similitud)")
    title: str
    pag: int

class ResponseLLM(BaseModel):
    answer: str
    evidence: Optional[List[ItemsEvidence]] =[]

class Messages(BaseModel):
    role: str
    content: str

class UserAsk(BaseModel):
    question: str
    chat_history: Optional[List[Messages]] = []