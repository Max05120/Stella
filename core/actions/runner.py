"""
core/actions/runner.py

Connects the action planner to Stella's existing capability system.

Flow:

    natural language
        ↓
    planner
        ↓
    ActionRequest
        ↓
    capability decision
        ↓
    executor / gap explanation / unknown
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ActionRequest
from .planner import plan_action

from core.capabilities import initialize_capabilities
from core.capabilities.decision import decide_capability
from core.capabilities.executor import execute_capability
from core.capabilities.registry import registry


@dataclass(slots=True)
class ActionRunResult:
    """
    High-level result returned by the action runner.
    """

    status: str
    message: str

    request: ActionRequest | None = None
    capability: str | None = None

    decision: Any | None = None
    execution: Any | None = None
    requires_confirmation: bool = False


def run_action(text: str) -> ActionRunResult:
    """
    Plan and execute a natural-language Mac action when possible.
    """
    normalized_text = text.strip().rstrip(".,!?;:")
    # ---------------------------------------------------------
    # 1. PLAN
    # ---------------------------------------------------------
    initialize_capabilities()
    planning = plan_action(normalized_text)

    if not planning.understood or planning.request is None:
        if _looks_like_action_request(normalized_text):
            return ActionRunResult(
                status="ACTION_NOT_UNDERSTOOD",
                message=(
                    "I understood that you're asking me to perform "
                    "a Mac action, but I couldn't safely determine "
                    "exactly what to do."
                ),
            )

        return ActionRunResult(
            status="NOT_UNDERSTOOD",
            message=(
                planning.reason
                or "The request could not be converted into an action."
            ),
        )
    request = planning.request

    # ---------------------------------------------------------
    # 2. CAPABILITY DECISION
    # ---------------------------------------------------------

    decision_query = _build_decision_query(request)
    print(f"[ACTION] decision query: {decision_query}")
    print(f"[ACTION] registered capabilities: {registry.names()}")
    decision = decide_capability(decision_query)

    decision_status = _decision_status(decision)

    # ---------------------------------------------------------
    # 3. AVAILABLE → EXECUTE
    # ---------------------------------------------------------

    if decision_status == "AVAILABLE":

        capability_name = _decision_capability_name(decision)
        capability = registry.get(
            capability_name
        )

        if not capability_name:
            return ActionRunResult(
                status="FAILED",
                message=(
                    "The capability decision marked the action as available "
                    "but did not provide a capability name."
                ),
                request=request,
                decision=decision,
            )

        execution = execute_capability(
            capability_name,
            request.arguments,
            # confirmed=True,
        )
        # print("[ACTION CONFIRM] execution:", execution)

        return ActionRunResult(
            status=_execution_status(execution),
            message=_execution_message(execution),
            request=request,
            capability=capability_name,
            decision=decision,
            execution=execution,
            requires_confirmation=(
                bool(
                    capability
                    and capability.requires_confirmation
                )
            ),
        )

    # ---------------------------------------------------------
    # 4. IMPLEMENTABLE
    # ---------------------------------------------------------

    if decision_status == "IMPLEMENTABLE":

        return ActionRunResult(
            status="IMPLEMENTABLE",
            message=(
                "Stella does not currently have an executable capability "
                "for this action, but the capability system believes it "
                "can be implemented."
            ),
            request=request,
            decision=decision,
        )

    # ---------------------------------------------------------
    # 5. BLOCKED
    # ---------------------------------------------------------

    if decision_status == "BLOCKED":

        return ActionRunResult(
            status="BLOCKED",
            message=(
                "The requested capability exists or is understood, "
                "but execution is currently blocked."
            ),
            request=request,
            decision=decision,
        )

    # ---------------------------------------------------------
    # 6. UNKNOWN
    # ---------------------------------------------------------

    return ActionRunResult(
        status="UNKNOWN",
        message=(
            "Stella could not identify a safe executable capability "
            "for this request."
        ),
        request=request,
        decision=decision,
    )


# ---------------------------------------------------------------------------
# Compatibility helpers
#
# These intentionally tolerate either dataclasses/enums or dictionaries.
# We can tighten these once runner.py is aligned with the exact structures
# currently returned by decision.py and executor.py.
# ---------------------------------------------------------------------------

ACTION_PREFIXES = {
    "open",
    "launch",
    "start",
    "show",
    "reveal",
    "create",
    "make",
    "move",
    "relocate",
    "copy",
    "duplicate",
    "rename",
    "delete",
    "remove",
    "trash",
    "list",
    "find",
}

def confirm_action(
    pending: ActionRunResult,
) -> ActionRunResult:
    """
    Execute an already-planned action after explicit user confirmation.

    This deliberately does not re-plan natural language.
    The exact capability and arguments from the pending action are reused.
    """

    request = pending.request
    capability_name = pending.capability

    if (
        request is None
        or not capability_name
    ):
        return ActionRunResult(
            status="FAILED",
            message=(
                "The pending action could not "
                "be confirmed safely."
            ),
        )

    capability = registry.get(
        capability_name
    )

    if capability is None:
        return ActionRunResult(
            status="FAILED",
            message=(
                "The pending capability is no "
                "longer available."
            ),
            request=request,
            capability=capability_name,
        )

    execution = execute_capability(
        capability_name,
        request.arguments,
        confirmed=True,
    )

    return ActionRunResult(
        status=_execution_status(
            execution
        ),
        message=_execution_message(
            execution
        ),
        request=request,
        capability=capability_name,
        decision=pending.decision,
        execution=execution,
        requires_confirmation=False,
    )

def _looks_like_action_request(
    text: str,
) -> bool:
    normalized = (
        text.strip()
        .lower()
    )

    if not normalized:
        return False

    first_word = (
        normalized
        .split(maxsplit=1)[0]
    )

    return first_word in ACTION_PREFIXES

def _decision_status(decision: Any) -> str:
    if decision is None:
        return "UNKNOWN"

    if isinstance(decision, dict):
        value = decision.get("status")
    else:
        value = getattr(decision, "status", None)

    if value is None:
        return "UNKNOWN"

    # Enum support
    value = getattr(value, "value", value)

    return str(value).upper()


def _decision_capability_name(decision: Any) -> str | None:
    if decision is None:
        return None

    if isinstance(decision, dict):
        capability = (
            decision.get("capability")
            or decision.get("capability_name")
        )
    else:
        capability = (
            getattr(decision, "capability", None)
            or getattr(decision, "capability_name", None)
        )

    if capability is None:
        return None

    # Decision layer may return a Capability object rather than its name.
    name = getattr(capability, "name", None)

    if name:
        return str(name)

    return str(capability)


def _execution_status(execution: Any) -> str:
    if execution is None:
        return "FAILED"

    if isinstance(execution, dict):
        value = execution.get("status")
    else:
        value = getattr(execution, "status", None)

    if value is None:
        return "FAILED"

    value = getattr(value, "value", value)

    return str(value).upper()


def _execution_message(execution: Any) -> str:
    if execution is None:
        return "Capability execution failed."

    if isinstance(execution, dict):
        message = execution.get("message")
    else:
        message = getattr(execution, "message", None)

    return message or "Capability execution completed."

def _build_decision_query(request: ActionRequest) -> str:
    """
    Convert a structured ActionRequest into the semantic capability query
    expected by the capability resolver.

    The target itself is intentionally excluded.

    Examples:

        Open TextEdit
        → open application

        Open github.com
        → open website

        Open ~/Desktop
        → open path

        Reveal ~/Desktop/test.txt in Finder
        → show file in finder

    The planner has already determined what kind of operation the user
    intends. The resolver's job is now to identify the matching executable
    capability, not to re-parse the original natural-language request.
    """

    if request.action == "open":
        if request.target_type == "application":
            return "open application"

        if request.target_type == "url":
            return "open website"

        if request.target_type == "path":
            return "open path"

    if request.action == "reveal" and request.target_type == "path":
        return "show file in finder"
    
    if (
        request.action == "trash"
        and request.target_type == "files"
    ):
        return "trash files"
    
    if (
        request.action == "trash"
        and request.target_type == "file"
    ):
        return "trash file"
    
    if (
        request.action == "create"
        and request.target_type == "folder"
    ):
        return "create folder"
    
    if (
        request.action == "move"
        and request.target_type == "file"
    ):
        return "move file"
    
    if (
        request.action == "move"
        and request.target_type == "files"
    ):
        return "move files"
    
    if (
        request.action == "copy"
        and request.target_type == "file"
    ):
        return "copy file"

    if (
        request.action == "copy"
        and request.target_type == "files"
    ):
        return "copy files"
    
    if (
        request.action == "rename"
        and request.target_type == "file"
    ):
        return "rename file"
    
    if (
        request.action == "list"
        and request.target_type == "files"
    ):
        return "list files"

    if (
        request.action == "inspect"
        and request.target_type == "finder_selection"
    ):
        return "get finder selection"
    if (
        request.action == "inspect"
        and request.target_type == "finder_selection"
    ):
        return "get finder selection"


    if (
        request.action == "inspect"
        and request.target_type == "finder_window_path"
    ):
        return "current finder folder"


    if (
        request.action == "inspect"
        and request.target_type == "finder_windows"
    ):
        return "get finder windows"


    if (
        request.action == "open"
        and request.target_type == "finder_location"
    ):
        return "open folder in finder"
    
    if (
        request.action == "inspect"
        and request.target_type == "frontmost_application"
    ):
        return "get frontmost application"


    if (
        request.action == "inspect"
        and request.target_type == "clipboard"
    ):
        return "get clipboard"


    if (
        request.action == "write"
        and request.target_type == "clipboard"
    ):
        return "write clipboard"


    if (
        request.action == "inspect"
        and request.target_type == "system_volume"
    ):
        return "get system volume"


    if (
        request.action == "set"
        and request.target_type == "system_volume"
    ):
        return "set_system_volume"


    if (
        request.action == "set"
        and request.target_type == "system_mute"
    ):
        return "mute_system_volume"


    if (
        request.action == "set"
        and request.target_type == "system_unmute"
    ):
        return "unmute_system_volume"
    if (
        request.action == "inspect"
        and request.target_type
        == "running_applications"
    ):
        return "get_running_applications"


    if (
        request.action == "activate"
        and request.target_type == "application"
    ):
        return "activate_application"


    if (
        request.action == "hide"
        and request.target_type == "application"
    ):
        return "hide_application"


    if (
        request.action == "quit"
        and request.target_type == "application"
    ):
        return "quit_application"


    if (
        request.action == "inspect"
        and request.target_type
        == "application_windows"
    ):
        return "get_application_windows"


    if (
        request.action == "inspect"
        and request.target_type
        == "frontmost_window"
    ):
        return "get_frontmost_window"


    if (
        request.action == "focus"
        and request.target_type == "window"
    ):
        return "focus_window"


    if (
        request.action == "move"
        and request.target_type == "window"
    ):
        return "move_window"


    if (
        request.action == "resize"
        and request.target_type == "window"
    ):
        return "resize_window"
    
    if (
        request.action == "inspect"
        and request.target_type == "focused_ui_element"
    ):
        return "get_focused_ui_element"


    if (
        request.action == "inspect"
        and request.target_type == "ui_elements"
    ):
        return "list_ui_elements"


    if (
        request.action == "click"
        and request.target_type == "ui_element"
    ):
        return "click_ui_element"


    if (
        request.action == "press"
        and request.target_type == "menu_item"
    ):
        return "press_menu_item"
    # Conservative fallback.
    #
    # Unknown structured combinations should still go through the decision
    # system rather than being mapped to an arbitrary capability.
    return f"{request.action} {request.target_type}"