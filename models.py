"""Typed models for SupportDesk RL Environment."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class SupportAction(BaseModel):
    """Action taken by the agent in the support desk environment."""
    action_type: str        # "classify" | "respond" | "escalate" | "close"
    value: str              # The classification label, response text, or escalation decision
    priority: Optional[str] = None   # "P0" | "P1" | "P2" | "P3" (for escalation task)
    rationale: Optional[str] = None  # Optional explanation from agent


class SupportObservation(BaseModel):
    """Observation returned to the agent after each step."""
    task_id: str                             # Which task is running
    item_id: str                             # Current ticket/incident ID
    content: str                             # The main text (ticket or incident description)
    context: Optional[str] = None           # Additional context (KB articles, logs, metrics)
    valid_actions: Optional[List[str]] = None  # Allowed action types for this step
    valid_values: Optional[List[str]] = None   # Allowed values (for classification tasks)
    step: int = 0                            # Current step in episode
    message: str = ""                        # Feedback from last action
    done: bool = False                       # Whether episode is over
    reward: float = 0.0                      # Reward from last action


class SupportState(BaseModel):
    """Episode state metadata."""
    task_id: str
    episode_id: str
    current_index: int = 0
    total_items: int = 0
    step_count: int = 0
    total_reward: float = 0.0
    done: bool = False
