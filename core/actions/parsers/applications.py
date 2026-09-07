"""
Deterministic application control parser.
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


def plan_application_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    )

    lower = text.lower()

    if lower in {
        "what apps are running",
        "what applications are running",
        "list running apps",
        "list running applications",
        "show running apps",
    }:
        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="running_applications",
            target="Running applications",
            arguments={},
        )

    match = re.match(
        r"^(?:switch to|activate|focus)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        application = (
            match.group(1).strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="activate",
            target_type="application",
            target=application,
            arguments={
                "application": application,
            },
        )

    match = re.match(
        r"^hide\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        application = (
            match.group(1).strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="hide",
            target_type="application",
            target=application,
            arguments={
                "application": application,
            },
        )

    match = re.match(
        r"^(?:quit|close|exit)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        application = (
            match.group(1).strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="quit",
            target_type="application",
            target=application,
            arguments={
                "application": application,
            },
        )

    return None