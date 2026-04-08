"""
Reward shaping for SupportDesk-OpenEnv.

Reward breakdown (max 1.0):
  +0.60  correct action type
  +0.20  response provided for respond_ticket actions
  +0.10  early resolution bonus (step <= 3)
  -0.10  latency penalty (step > 10)
  -0.20  hard penalty (step > 15, runaway loop)
"""

from typing import Optional


CORRECT_ACTION_REWARD = 0.60
RESPONSE_BONUS = 0.20
EARLY_BONUS = 0.10
LATE_PENALTY_SOFT = 0.10
LATE_PENALTY_HARD = 0.20


def compute_reward(
    action_type: str,
    correct_action: str,
    step: int,
    response: Optional[str] = None,
) -> float:
    reward = 0.0

    # Core signal: did the agent choose the right action?
    if action_type == correct_action:
        reward += CORRECT_ACTION_REWARD

    # Bonus: agent provided a written response when resolving
    if action_type == "respond_ticket" and response and len(response.strip()) > 10:
        reward += RESPONSE_BONUS

    # Bonus: solved quickly (within first 3 steps of an episode)
    if action_type == correct_action and step <= 3:
        reward += EARLY_BONUS

    # Soft latency penalty
    if step > 10:
        reward -= LATE_PENALTY_SOFT

    # Hard latency penalty — agent is looping
    if step > 15:
        reward -= LATE_PENALTY_HARD

    return round(max(0.0, min(reward, 1.0)), 4)
