"""
resolver.py

Maps a requested intent/action to capabilities that Stella
currently has registered.

This is deliberately separate from the planner.

Planner:
    "What does the user want?"

Resolver:
    "Which of Stella's available capabilities can satisfy it?"
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

from core.capabilities.models import Capability
from core.capabilities.registry import registry


@dataclass
class CapabilityMatch:
    capability: Capability
    score: float
    matched_text: str


def _normalize(text: str) -> str:
    return " ".join(
        text.lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _similarity(
    left: str,
    right: str,
) -> float:
    """
    Basic lexical similarity.

    This is intentionally simple for the first version.
    Later we'll replace/augment it with embedding or LLM-based
    semantic resolution.
    """

    left = _normalize(left)
    right = _normalize(right)

    if not left or not right:
        return 0.0

    # Exact match.
    if left == right:
        return 1.0

    # Strong partial match.
    if left in right or right in left:
        return 0.9

    return SequenceMatcher(
        None,
        left,
        right,
    ).ratio()


def _capability_search_terms(
    capability: Capability,
) -> list[str]:
    """
    Build all strings that can identify this capability.
    """

    terms = [
        capability.name,
        capability.description,
    ]

    terms.extend(
        capability.intents
    )

    return terms

ACTION_GROUPS = {
    "open": {
        "open",
        "launch",
        "start",
        "show",
        "reveal",
    },

    "delete": {
        "delete",
        "remove",
        "trash",
        "erase",
    },

    "move": {
        "move",
        "relocate",
    },

    "copy": {
        "copy",
        "duplicate",
    },

    "create": {
        "create",
        "make",
        "new",
    },

    "close": {
        "close",
        "quit",
        "exit",
    },
}

ACTION_ALIASES = {
    "open": {
        "open",
        "launch",
        "start",
        "reveal",
        "show",
    },

    "close": {
        "close",
        "quit",
        "exit",
        "terminate",
    },

    "click": {
        "click",
        "press",
        "tap",
    },

    "delete": {
        "delete",
        "remove",
        "trash",
        "erase",
    },

    "move": {
        "move",
        "relocate",
    },

    "copy": {
        "copy",
        "duplicate",
    },

    "capture": {
        "capture",
        "screenshot",
        "record",
    },

    "run": {
        "run",
        "execute",
    },

    "watch": {
        "watch",
        "monitor",
        "observe",
    },

    "create": {
        "create",
        "make",
        "new",
    },
}

def _detect_action(
    text: str,
) -> str | None:

    words = set(
        _normalize(text).split()
    )

    for action, aliases in ACTION_ALIASES.items():

        if words.intersection(aliases):
            return action

    return None

def resolve_capabilities(
    query: str,
    limit: int = 5,
    minimum_score: float = 0.45,
) -> list[CapabilityMatch]:
    """
    Return matching registered capabilities,
    highest score first.
    """

    matches: list[CapabilityMatch] = []

    query_action = _detect_action(query)

    for capability in registry.all():

        best_score = 0.0
        best_text = ""

        for term in _capability_search_terms(
            capability
        ):

            score = _similarity(
                query,
                term,
            )

            if score > best_score:
                best_score = score
                best_text = term
        capability_terms = " ".join(
            _capability_search_terms(
                capability
            )
        )

        capability_action = _detect_action(
            capability_terms
        )

        if (
            query_action
            and capability.action
            and query_action != capability.action
        ):
            continue

        if best_score >= minimum_score:

            matches.append(
                CapabilityMatch(
                    capability=capability,
                    score=best_score,
                    matched_text=best_text,
                )
            )

    matches.sort(
        key=lambda match: match.score,
        reverse=True,
    )

    return matches[:limit]


def resolve_best_capability(
    query: str,
    minimum_score: float = 0.72,
) -> CapabilityMatch | None:
    """
    Return the strongest available capability match.
    """

    matches = resolve_capabilities(
        query=query,
        limit=1,
        minimum_score=minimum_score,
    )

    if not matches:
        return None

    return matches[0]