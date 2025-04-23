from typing import Optional
from pydantic import BaseModel

class DataColumnModel(BaseModel):
    questionColumn: str
    answerColumn: str
    contextColumn: Optional[str] = None
    systemPrompt: Optional[str] = None