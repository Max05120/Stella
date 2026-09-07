"""
Deterministic Accessibility/UI parser.
"""

from __future__ import annotations

import re

from core.actions.models import (
    ActionRequest,
)

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


def plan_accessibility_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    )

    lower = text.lower()

    # -----------------------------------------------------
    # Focused UI element
    # -----------------------------------------------------

    if lower in {
        "what is focused",
        "what's focused",
        "what ui element is focused",
        "what control is focused",
        "what field is focused",
        "get focused ui element",
    }:
        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="focused_ui_element",
            target="Focused UI element",
            arguments={},
        )

    # -----------------------------------------------------
    # UI inspection
    # -----------------------------------------------------

    match = re.match(
        r"^(?:list|show|inspect)\s+"
        r"(?:the\s+)?(?:ui|controls|ui elements)\s+"
        r"(?:in|for|of)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        application = (
            match.group(1)
            .strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="ui_elements",
            target=application,
            arguments={
                "application": application,
            },
        )
    # -----------------------------------------------------
    # Menu item
    # -----------------------------------------------------

    match = re.match(
        r"^(?:click|press|choose)\s+"
        r"(.+?)\s+"
        r"(?:from|in)\s+"
        r"(?:the\s+)?(.+?)\s+menu\s+"
        r"(?:in|on)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        menu_item = (
            match.group(1)
            .strip()
        )

        menu_name = (
            match.group(2)
            .strip()
        )

        application = (
            match.group(3)
            .strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="press",
            target_type="menu_item",
            target=menu_item,
            arguments={
                "application": application,
                "menu_name": menu_name,
                "menu_item": menu_item,
            },
        )

    # -----------------------------------------------------
    # Click UI element
    # -----------------------------------------------------

    patterns = [
        r"^(?:click|press)\s+"
        r"(?:the\s+)?(.+?)\s+"
        r"(?:button|control)\s+"
        r"(?:in|on)\s+(.+)$",

        r"^(?:click|press)\s+"
        r"(.+?)\s+in\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        label = (
            match.group(1)
            .strip()
        )

        application = (
            match.group(2)
            .strip()
        )

        return ActionRequest(
            raw_text=raw_text,
            action="click",
            target_type="ui_element",
            target=label,
            arguments={
                "application": application,
                "label": label,
            },
        )
    
    return None