"""
Deterministic workspace action parser.

Workspace actions include:

- open application
- open URL
- open path
- open common macOS folder
- reveal path in Finder
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from core.actions.models import ActionRequest
from core.actions.parsers.common import (
    SPECIAL_FOLDERS,
    clean_path,
    looks_like_path,
    strip_polite_prefixes,
)


def plan_workspace_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    """
    Try workspace parsers in conservative order.
    """

    result = _plan_reveal_in_finder(
        raw_text,
        normalized,
    )

    if result:
        return result

    result = _plan_url(
        raw_text,
        normalized,
    )

    if result:
        return result

    result = _plan_path(
        raw_text,
        normalized,
    )

    if result:
        return result

    result = _plan_special_folder(
        raw_text,
        normalized,
    )

    if result:
        return result

    result = _plan_application(
        raw_text,
        normalized,
    )

    if result:
        return result

    return None


def _plan_special_folder(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^open\s+(?:my\s+)?(.+?)(?:\s+folder)?$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    folder_name = (
        match.group(1)
        .strip()
        .lower()
    )

    path = SPECIAL_FOLDERS.get(
        folder_name
    )

    if path is None:
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="open",
        target_type="path",
        target=str(path),
        arguments={
            "path": str(path),
        },
    )


def _plan_url(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^(?:open|visit|go to|launch)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    candidate = match.group(1).strip()

    url = _normalize_url(
        candidate
    )

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


def _normalize_url(
    value: str,
) -> str | None:
    value = value.strip()

    parsed = urlparse(value)

    if (
        parsed.scheme in {"http", "https"}
        and parsed.netloc
    ):
        return value

    if re.match(
        r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/.*)?$",
        value,
    ):
        return f"https://{value}"

    return None


def _plan_reveal_in_finder(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

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

        path = clean_path(
            match.group(1)
        )

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


def _plan_path(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^open\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    candidate = clean_path(
        match.group(1)
    )

    if not looks_like_path(
        candidate
    ):
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


def _plan_application(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^(?:open|launch|start)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    application = (
        match.group(1)
        .strip()
    )

    if not application:
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="open",
        target_type="application",
        target=application,
        arguments={
            "application": application,
        },
    )