"""
Deterministic parser for system context and basic system controls.
"""

from __future__ import annotations

import re

from core.actions.models import (
    ActionRequest,
)

from core.actions.parsers.common import (
    strip_polite_prefixes,
)


def plan_system_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    planners = [
        _plan_frontmost_application,
        _plan_get_clipboard,
        _plan_set_clipboard,
        _plan_get_volume,
        _plan_set_volume,
        _plan_unmute,
        _plan_mute,
    ]

    for planner in planners:

        result = planner(
            raw_text,
            normalized,
        )

        if result:
            return result

    return None


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


def _plan_frontmost_application(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    ).lower()

    patterns = [
        r"^what(?:'s| is)\s+the\s+current\s+app$",
        r"^what(?:'s| is)\s+the\s+active\s+app$",
        r"^what\s+app\s+is\s+active$",
        r"^what\s+application\s+is\s+active$",
        r"^what(?:'s| is)\s+the\s+frontmost\s+application$",
        r"^what\s+app\s+am\s+i\s+using$",
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
                target_type="frontmost_application",
                target="Frontmost application",
                arguments={},
            )

    return None


def _plan_get_clipboard(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    ).lower()

    patterns = [
        r"^what(?:'s| is)\s+in\s+(?:my\s+)?clipboard$",
        r"^read\s+(?:my\s+)?clipboard$",
        r"^show\s+me\s+(?:my\s+)?clipboard$",
        r"^what\s+did\s+i\s+copy$",
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
                target_type="clipboard",
                target="Clipboard",
                arguments={},
            )

    return None


def _plan_set_clipboard(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    )

    patterns = [
        r"^copy\s+(.+?)\s+to\s+(?:my\s+)?clipboard$",
        r"^put\s+(.+?)\s+(?:on|in)\s+(?:my\s+)?clipboard$",
        r"^set\s+(?:my\s+)?clipboard\s+to\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = (
            match.group(1)
            .strip()
        )

        if not value:
            return None

        return ActionRequest(
            raw_text=raw_text,
            action="write",
            target_type="clipboard",
            target=value,
            arguments={
                "text": value,
            },
        )

    return None


def _plan_get_volume(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    ).lower()

    patterns = [
        r"^what(?:'s| is)\s+(?:the\s+)?volume$",
        r"^what(?:'s| is)\s+(?:the\s+)?current\s+volume$",
        r"^what\s+is\s+(?:the\s+)?volume\s+level$",
        r"^get\s+(?:the\s+)?volume$",
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
                target_type="system_volume",
                target="System volume",
                arguments={},
            )

    return None


def _plan_set_volume(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    )

    patterns = [
        r"^set\s+(?:the\s+)?volume\s+to\s+(\d{1,3})(?:\s*%)?$",
        r"^change\s+(?:the\s+)?volume\s+to\s+(\d{1,3})(?:\s*%)?$",
        r"^volume\s+(\d{1,3})(?:\s*%)?$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        volume = int(
            match.group(1)
        )

        if volume < 0 or volume > 100:
            return None

        return ActionRequest(
            raw_text=raw_text,
            action="set",
            target_type="system_volume",
            target=str(volume),
            arguments={
                "volume": volume,
            },
        )

    return None


def _plan_mute(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    ).lower()

    if text in {
        "mute",
        "mute volume",
        "mute sound",
        "mute audio",
        "mute the volume",
        "mute the sound",
    }:
        return ActionRequest(
            raw_text=raw_text,
            action="set",
            target_type="system_mute",
            target="mute",
            arguments={},
        )

    return None


def _plan_unmute(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    text = _clean(
        normalized
    ).lower()

    if text in {
        "unmute",
        "unmute volume",
        "unmute sound",
        "unmute audio",
        "unmute the volume",
        "unmute the sound",
        "turn sound back on",
        "turn the sound back on",
    }:
        return ActionRequest(
            raw_text=raw_text,
            action="set",
            target_type="system_unmute",
            target="unmute",
            arguments={},
        )

    return None