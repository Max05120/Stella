"""
Deterministic macOS window parser.
"""

from __future__ import annotations

import re

from core.actions.models import ActionRequest
from core.actions.parsers.common import (
    strip_polite_prefixes,
)


def _clean(
    normalized: str,
) -> str:
    return (
        strip_polite_prefixes(
            normalized
        )
        .strip()
        .rstrip(".,!?;:")
        .strip()
    )


def plan_window_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    )

    lower = text.lower()

    if lower in {
        "what is the current window",
        "what's the current window",
        "what is the active window",
        "what's the active window",
        "what is the frontmost window",
        "what's the frontmost window",
    }:
        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="frontmost_window",
            target="Frontmost window",
            arguments={},
        )

    match = re.match(
        r"^(?:list|show|get)\s+(.+?)\s+windows$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        application = (
            match.group(1).strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="application_windows",
            target=application,
            arguments={
                "application": application,
            },
        )

    match = re.match(
        r"^focus\s+(.+?)\s+window(?:\s+(\d+))?$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        application = (
            match.group(1).strip()
        )

        index = (
            int(match.group(2)) - 1
            if match.group(2)
            else 0
        )

        return ActionRequest(
            raw_text=raw_text,
            action="focus",
            target_type="window",
            target=application,
            arguments={
                "application": application,
                "window_index": index,
            },
        )

    match = re.match(
        r"^move\s+(.+?)\s+window\s+to\s+"
        r"(\d+)\s*,?\s*(\d+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return ActionRequest(
            raw_text=raw_text,
            action="move",
            target_type="window",
            target=match.group(1).strip(),
            arguments={
                "application":
                    match.group(1).strip(),
                "x":
                    int(match.group(2)),
                "y":
                    int(match.group(3)),
                "window_index": 0,
            },
        )

    match = re.match(
        r"^resize\s+(.+?)\s+window\s+to\s+"
        r"(\d+)\s*[x×]\s*(\d+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return ActionRequest(
            raw_text=raw_text,
            action="resize",
            target_type="window",
            target=match.group(1).strip(),
            arguments={
                "application":
                    match.group(1).strip(),
                "width":
                    int(match.group(2)),
                "height":
                    int(match.group(3)),
                "window_index": 0,
            },
        )

    return None