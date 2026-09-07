"""
Deterministic Finder parser.

Supports:
- current Finder selection
- front Finder window path
- open Finder windows
- opening a location in Finder
"""

from __future__ import annotations

import re

from core.actions.models import (
    ActionRequest,
)

from core.actions.parsers.common import (
    resolve_folder_location,
    strip_polite_prefixes,
)

print("[DEBUG] finder parser module loaded")

def plan_finder_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    print(
    "[DEBUG] plan_finder_action called:",
    repr(normalized),
    )
    text = strip_polite_prefixes(
        normalized
    ).strip()

    lowered = (
        text.lower()
        .rstrip(".,!?;:")
        .strip()
    )

    # ---------------------------------------------------------
    # Explicit Finder-window inspection
    # ---------------------------------------------------------

    if lowered in {
        "what finder windows are open",
        "list the open finder windows",
        "list finder windows",
        "show finder windows",
        "show me the finder windows",
        "show me the open finder windows",
        "which finder windows are open",
    }:
        print("[FINDER PARSER] matched finder windows")
        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="finder_windows",
            target="Finder windows",
            arguments={},
        )

    planners = [
        _plan_finder_selection,
        _plan_finder_window_path,
        _plan_finder_windows,
        _plan_open_finder_location,
    ]

    for planner in planners:
        result = planner(
            raw_text,
            normalized,
        )

        if result:
            return result

    return None


def _plan_finder_selection(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = strip_polite_prefixes(
        normalized
    ).strip()

    patterns = [
        r"^what\s+do\s+i\s+have\s+selected(?:\s+in\s+finder)?$",

        r"^what(?:'s| is)\s+selected(?:\s+in\s+finder)?$",

        r"^what(?:'s| is)\s+the\s+selected\s+(?:file|files|item|items)$",

        r"^show\s+me\s+the\s+selected\s+(?:file|files|item|items)$",

        r"^show\s+me\s+what(?:'s| is)\s+selected(?:\s+in\s+finder)?$",

        r"^get\s+(?:the\s+)?finder\s+selection$",
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            return ActionRequest(
                raw_text=raw_text,
                action="inspect",
                target_type="finder_selection",
                target="Finder selection",
                arguments={},
            )

    return None


def _plan_finder_window_path(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = strip_polite_prefixes(
        normalized
    ).strip()

    patterns = [
        r"^where\s+am\s+i\s+in\s+finder$",

        r"^what\s+folder\s+am\s+i\s+in(?:\s+in\s+finder)?$",

        r"^what(?:'s| is)\s+the\s+current\s+finder\s+(?:folder|location|path)$",

        r"^get\s+(?:the\s+)?finder\s+(?:folder|location|path)$",

        r"^what(?:'s| is)\s+the\s+front\s+finder\s+window(?:\s+path)?$",
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            return ActionRequest(
                raw_text=raw_text,
                action="inspect",
                target_type="finder_window_path",
                target="Finder window path",
                arguments={},
            )

    return None


def _plan_finder_windows(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = strip_polite_prefixes(
        normalized
    ).strip()

    lowered = text.lower()

    patterns = [
        r"^what\s+finder\s+windows\s+are\s+open$",
        r"^which\s+finder\s+windows\s+are\s+open$",

        r"^show\s+me\s+(?:the\s+)?open\s+finder\s+windows$",

        r"^show\s+me\s+(?:the\s+)?finder\s+windows$",

        r"^list\s+(?:the\s+)?finder\s+windows$",

        r"^list\s+(?:the\s+)?open\s+finder\s+windows$",

        r"^what\s+windows\s+are\s+open\s+in\s+finder$",
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return ActionRequest(
                raw_text=raw_text,
                action="inspect",
                target_type="finder_windows",
                target="Finder windows",
                arguments={},
            )

    # Conservative semantic fallback for this
    # particular deterministic intent.
    if (
        "finder" in lowered
        and "window" in lowered
        and (
            "open" in lowered
            or "list" in lowered
            or "show" in lowered
        )
    ):
        return ActionRequest(
            raw_text=raw_text,
            action="inspect",
            target_type="finder_windows",
            target="Finder windows",
            arguments={},
        )

    return None


def _plan_open_finder_location(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = strip_polite_prefixes(
        normalized
    ).strip()

    patterns = [
        r"^open\s+(.+?)\s+in\s+finder$",

        r"^show\s+(.+?)\s+in\s+finder$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        path = resolve_folder_location(
            match.group(1)
        )

        if path is None:
            return None

        return ActionRequest(
            raw_text=raw_text,
            action="open",
            target_type="finder_location",
            target=path,
            arguments={
                "path": path,
            },
        )

    return None