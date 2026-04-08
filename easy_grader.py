"""
Easy grader — Ticket Classification.
Binary: correct classification = 1.0, incorrect = 0.0.
"""


def grade(action: str, expected: str) -> float:
    return 1.0 if action.strip().lower() == expected.strip().lower() else 0.0
