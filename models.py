from pydantic import BaseModel, Field
from typing import List, Optional, Literal


ActionType = Literal["classify_ticket", "respond_ticket", "escalate_ticket"]


class Action(BaseModel):
    action_type: ActionType
    ticket_id: str
    response: Optional[str] = None
    reasoning: Optional[str] = None


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


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class EpisodeStats(BaseModel):
    episode: int
    total_reward: float
    steps: int
    success: bool
    tickets_correct: int
    tickets_total: int
