"""
Explicit confirmation recognition for pending actions.

This module only interprets confirmation/cancellation language.
It does not execute capabilities.
"""

from __future__ import annotations


CONFIRM_PHRASES = {
    "yes",
    "yes please",
    "yeah",
    "yep",
    "confirm",
    "confirmed",
    "do it",
    "go ahead",
    "proceed",
}

CANCEL_PHRASES = {
    "no",
    "nope",
    "cancel",
    "stop",
    "don't",
    "do not",
    "never mind",
    "nevermind",
}


def confirmation_intent(
    text: str,
) -> bool | None:
    """
    Return:

        True  → explicit confirmation
        False → explicit cancellation
        None  → not a confirmation response
    """

    normalized = (
        " ".join(
            text.lower()
            .strip()
            .split()
        )
    )

    normalized = normalized.rstrip(
        ".!?"
    )

    if normalized in CONFIRM_PHRASES:
        return True

    if normalized in CANCEL_PHRASES:
        return False

    return None