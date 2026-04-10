from pydantic import BaseModel, Field
from typing import List, Literal

ActionType = Literal["classify_ticket", "respond_ticket", "escalate_ticket"]

class Ticket(BaseModel):
    id: str
    text: str
    category: str
    priority: str
    correct_action: ActionType
    expected_category: str
    solution_keywords: List[str]

class Observation(BaseModel):
    ticket: Ticket
    step: int
    episode: int
    knowledge_base: List[str] = Field(default_factory=list)
