"""
core/actions/planner.py

Deterministic v1 planner.

This planner intentionally supports only the workspace capabilities
that Stella can already execute.

It does NOT attempt to understand arbitrary Mac commands.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import ActionRequest, PlanningResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_action(text: str) -> PlanningResult:
    """
    Convert natural-language text into a structured ActionRequest.

    The planner is conservative by design. If it cannot confidently
    understand the request, it returns PlanningResult.failure().
    """

    if not text or not text.strip():
        return PlanningResult.failure("Request is empty.")

    raw_text = text.strip()
    normalized = _normalize(raw_text)

    # Order matters.
    #
    # "Show /path in Finder" must be checked before generic path opening,
    # otherwise it could incorrectly become open_path.

    result = _plan_reveal_in_finder(raw_text, normalized)
    if result:
        return PlanningResult.success(result)

    result = _plan_url(raw_text, normalized)
    if result:
        return PlanningResult.success(result)

    result = _plan_path(raw_text, normalized)
    if result:
        return PlanningResult.success(result)

    result = _plan_application(raw_text, normalized)
    if result:
        return PlanningResult.success(result)

    return PlanningResult.failure(
        "The deterministic planner does not understand this action yet."
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = text.strip()

    # Collapse repeated whitespace without modifying paths/URLs more
    # aggressively than necessary.
    text = re.sub(r"\s+", " ", text)

    return text


def _strip_polite_prefixes(text: str) -> str:
    """
    Remove lightweight conversational prefixes without trying to perform
    general NLP.
    """

    patterns = [
        r"^please\s+",
        r"^can you\s+",
        r"^could you\s+",
        r"^would you\s+",
        r"^stella[,\s]+",
        r"^hey stella[,\s]+",
    ]

    result = text.strip()

    changed = True
    while changed:
        changed = False

        for pattern in patterns:
            updated = re.sub(
                pattern,
                "",
                result,
                flags=re.IGNORECASE,
            )

            if updated != result:
                result = updated.strip()
                changed = True

    return result


# ---------------------------------------------------------------------------
# URL planning
# ---------------------------------------------------------------------------


def _plan_url(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _strip_polite_prefixes(normalized)

    match = re.match(
        r"^(?:open|visit|go to|launch)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    candidate = match.group(1).strip()

    url = _normalize_url(candidate)

    if url is None:
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="open",
        target_type="url",
        target=url,
        arguments={
            "url": url,
        },
    )


def _normalize_url(value: str) -> str | None:
    value = value.strip()

    parsed = urlparse(value)

    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value

    # Handle things like:
    # open google.com
    # open github.com
    if re.match(
        r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/.*)?$",
        value,
    ):
        return f"https://{value}"

    return None


# ---------------------------------------------------------------------------
# Finder reveal planning
# ---------------------------------------------------------------------------


def _plan_reveal_in_finder(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _strip_polite_prefixes(normalized)

    patterns = [
        r"^(?:show|reveal)\s+(.+?)\s+in\s+finder$",
        r"^show\s+(.+?)\s+in\s+the\s+finder$",
        r"^reveal\s+(.+?)\s+in\s+the\s+finder$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        path = _clean_path(match.group(1))

        if not path:
            return None

        return ActionRequest(
            raw_text=raw_text,
            action="reveal",
            target_type="path",
            target=path,
            arguments={
                "path": path,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Path planning
# ---------------------------------------------------------------------------


def _plan_path(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _strip_polite_prefixes(normalized)

    match = re.match(
        r"^(?:open)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    candidate = _clean_path(match.group(1))

    if not _looks_like_path(candidate):
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="open",
        target_type="path",
        target=candidate,
        arguments={
            "path": candidate,
        },
    )


def _clean_path(value: str) -> str:
    value = value.strip()

    # Allow:
    # open "/Users/max/foo.txt"
    # open '/Users/max/foo.txt'
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]

    return value.strip()


def _looks_like_path(value: str) -> bool:
    if not value:
        return False

    return (
        value.startswith("/")
        or value.startswith("~/")
        or value.startswith("./")
        or value.startswith("../")
    )


# ---------------------------------------------------------------------------
# Application planning
# ---------------------------------------------------------------------------


def _plan_application(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _strip_polite_prefixes(normalized)

    match = re.match(
        r"^(?:open|launch|start)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    application = match.group(1).strip()

    if not application:
        return None

    # By the time we reach this function:
    # - URLs have already been handled
    # - explicit paths have already been handled
    #
    # Therefore the remaining target can conservatively be interpreted
    # as an application for the narrow v1 planner.

    return ActionRequest(
        raw_text=raw_text,
        action="open",
        target_type="application",
        target=application,
        arguments={
            "application": application,
        },
    )