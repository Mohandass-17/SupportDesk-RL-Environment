"""
Hard grader — Escalation Decision.
Escalation errors are costly: failing to escalate a critical ticket = 0.0.
Correct escalation = 1.0. Wrong action = 0.1 (no partial credit).
"""


def grade(action: str, expected: str) -> float:
    if action.strip().lower() == expected.strip().lower():
        return 1.0
    # Escalation misses are high-severity failures
    return 0.1
