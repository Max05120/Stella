"""
core/actions/planner.py

Top-level deterministic action planner.

This module does not contain family-specific parsing logic.
It delegates to specialized parser families.
"""

from __future__ import annotations

from core.actions.models import (
    PlanningResult,
)
from core.actions.parsers.common import (
    normalize,
)
from core.actions.parsers.filesystem import (
    plan_filesystem_action,
)
from core.actions.parsers.workspace import (
    plan_workspace_action,
)
from core.actions.parsers.discovery import (
    plan_discovery_action,
)
from core.actions.parsers.finder import (
    plan_finder_action,
)
from core.actions.parsers.system import (
    plan_system_action,
)
from core.actions.parsers.applications import (
    plan_application_action,
)

from core.actions.parsers.windows import (
    plan_window_action,
)
from core.actions.parsers.accessibility import (
    plan_accessibility_action,
)

PARSER_FAMILIES = [
    plan_filesystem_action,
    plan_finder_action,
    plan_discovery_action,
    plan_system_action,
    plan_window_action,
    plan_accessibility_action,
    plan_application_action,
    plan_workspace_action,
]


def plan_action(
    text: str,
) -> PlanningResult:
    """
    Convert natural-language text into a structured ActionRequest.

    Parser families remain conservative and deterministic.
    """

    if not text or not text.strip():
        return PlanningResult.failure(
            "Request is empty."
        )

    raw_text = text.strip()

    normalized = normalize(
        raw_text
    )
    print(
    "[DEBUG] plan_finder_action called:",
    repr(normalized),
    )
    for parser in PARSER_FAMILIES:
        request = parser(
            raw_text,
            normalized,
        )

        if request is not None:
            return PlanningResult.success(
                request
            )

    return PlanningResult.failure(
        "The deterministic planner does not understand "
        "this action yet."
    )