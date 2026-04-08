"""
Medium grader — Ticket Resolution.
Full credit for correct action with a non-empty response.
Partial credit for correct action with no response.
Zero for wrong action.
"""


def grade(action: str, expected: str, response: str = "") -> float:
    if action.strip().lower() != expected.strip().lower():
        return 0.0
    # Correct action: bonus for providing an actual response
    if response and len(response.strip()) > 10:
        return 1.0
    return 0.5
