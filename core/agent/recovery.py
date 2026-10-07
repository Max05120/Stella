"""Deterministic failure categories, bounded retry policy, and explanations."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.models import AgentObservation

# Reviewed inspection handlers. Names/prefixes alone never establish safety.
READ_ONLY_CAPABILITIES = frozenset({
    "get_system_volume", "get_frontmost_application", "get_clipboard",
    "get_running_applications", "list_files", "get_finder_selection",
    "get_finder_window_path", "get_finder_windows", "get_application_windows",
    "get_frontmost_window", "get_focused_ui_element", "list_ui_elements",
})


class RecoveryType(str, Enum):
    ALREADY_EXISTS = "already_exists"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    INVALID_ARGUMENT = "invalid_argument"
    INVALID_STATE = "invalid_state"
    TOOL_UNAVAILABLE = "tool_unavailable"
    APPLICATION_UNAVAILABLE = "application_unavailable"
    TIMEOUT = "timeout"
    PLANNER_FAILURE = "planner_failure"
    CONFIRMATION_REQUIRED = "confirmation_required"
    TRANSIENT = "transient"
    NON_RETRYABLE = "non_retryable"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RecoveryAssessment:
    recovery_type: RecoveryType
    retry_same_action: bool
    guidance: str


_GUIDANCE = {
    RecoveryType.ALREADY_EXISTS: "The resource already exists. Inspect or use it only if it meets the original goal; do not overwrite it or assume later subgoals are complete.",
    RecoveryType.NOT_FOUND: "Locate the resource with a registered inspection capability or ask for the correct target. Do not repeat the identical missing-resource call.",
    RecoveryType.PERMISSION_DENIED: "Permission was denied. Do not bypass the restriction using another capability. Explain the required permission or ask the user to resolve it.",
    RecoveryType.INVALID_ARGUMENT: "Correct arguments against the registered capability schema. Do not repeat the identical invalid call.",
    RecoveryType.INVALID_STATE: "Inspect the current application or resource state. Choose a valid strategy within the original goal, or ask the user to resolve the state.",
    RecoveryType.TOOL_UNAVAILABLE: "This capability is unavailable. Choose another registered capability only if it can satisfy the same requirement safely; otherwise explain the missing capability.",
    RecoveryType.APPLICATION_UNAVAILABLE: "The required application is unavailable. Inspect running applications or ask the user to make it available; do not repeatedly dispatch the same operation.",
    RecoveryType.TIMEOUT: "The operation timed out. A state-changing operation may already have taken effect. Inspect the outcome; do not repeat or replace an uncertain mutation automatically.",
    RecoveryType.PLANNER_FAILURE: "The planner could not produce a usable decision within its repair allowance. Stop with the saved progress; do not dispatch an unvalidated action.",
    RecoveryType.CONFIRMATION_REQUIRED: "Wait for explicit confirmation. Do not retry or substitute another capability to bypass approval.",
    RecoveryType.TRANSIENT: "The failure appears temporary. A retry must still advance the original goal and obey the remaining run limits.",
    RecoveryType.NON_RETRYABLE: "This failure is not retryable. Use a supported alternative only when its safety and relevance are established, or explain why the goal is blocked.",
    RecoveryType.UNKNOWN: "The cause is uncertain. Inspect existing evidence or ask the user. Do not repeat the failed call or assume a mutation had no effect.",
}


def assess_failure(observation: "AgentObservation", *, exception=None) -> RecoveryAssessment:
    status = str(observation.status).lower()
    text = " ".join(str(v) for v in (observation.error, observation.message) if v).lower()

    def contains(*phrases):
        return any(p in text for p in phrases)

    if observation.requires_confirmation and status == "blocked":
        category = RecoveryType.CONFIRMATION_REQUIRED
    elif isinstance(exception, PermissionError) or contains("permissionerror:", "permission denied", "access denied", "not permitted", "not authorized", "not authorised", "-1743"):
        category = RecoveryType.PERMISSION_DENIED
    elif isinstance(exception, TimeoutError) or contains("timeouterror:", "timed out", "timeout", "-1712"):
        category = RecoveryType.TIMEOUT
    elif status == "not_found" or contains("capability is not registered", "unknown capability", "tool unavailable", "tool is unavailable") or ("capability" in text and contains("not currently available", "not registered")):
        category = RecoveryType.TOOL_UNAVAILABLE
    elif contains("application not running", "application is not running", "application unavailable", "application not found", "application is unavailable", "can't get application", "cannot find application", "application isn't running", "-600"):
        category = RecoveryType.APPLICATION_UNAVAILABLE
    elif isinstance(exception, FileExistsError) or contains("fileexistserror:", "already exists", "file exists"):
        category = RecoveryType.ALREADY_EXISTS
    elif isinstance(exception, FileNotFoundError) or contains("filenotfounderror:", "no such file", "does not exist", "could not find", "not found"):
        category = RecoveryType.NOT_FOUND
    elif contains("invalid capability arguments", "invalid argument", "unexpected keyword", "required positional") or ("missing" in text and "argument" in text):
        category = RecoveryType.INVALID_ARGUMENT
    elif isinstance(exception, (NotADirectoryError, IsADirectoryError)) or contains("invalid state", "not a directory", "is a directory", "no active window", "no focused window", "no selection", "stale element"):
        category = RecoveryType.INVALID_STATE
    elif isinstance(exception, NotImplementedError) or contains("notimplementederror:", "unsupported operation", "not supported"):
        category = RecoveryType.NON_RETRYABLE
    elif isinstance(exception, ConnectionError) or contains("connectionerror:", "connectionreseterror:", "connectionrefusederror:", "temporarily unavailable", "connection reset", "connection refused", "try again", "rate limit"):
        category = RecoveryType.TRANSIENT
    else:
        # Existing trusted adapters may supply classification without text.
        try:
            category = RecoveryType(observation.recovery_type)
        except (ValueError, TypeError):
            category = RecoveryType.TRANSIENT if observation.retry_same_action is True else RecoveryType.UNKNOWN

    retry = (
        category in {RecoveryType.TRANSIENT, RecoveryType.TIMEOUT}
        and observation.capability in READ_ONLY_CAPABILITIES
        and observation.retry_same_action is not False
    )
    guidance = _GUIDANCE[category]
    if retry:
        guidance += " At most one identical retry of this reviewed read-only capability is allowed; it consumes the existing budgets."
    elif category in {RecoveryType.TRANSIENT, RecoveryType.TIMEOUT}:
        guidance += " This call is not eligible for an identical automatic retry."
    return RecoveryAssessment(category, retry, guidance)


def annotate_failure(observation: "AgentObservation", *, exception=None):
    if observation.success:
        observation.recovery_type = None
        observation.retry_same_action = None
        observation.recovery_guidance = None
        return observation
    assessment = assess_failure(observation, exception=exception)
    observation.recovery_type = assessment.recovery_type.value
    observation.retry_same_action = assessment.retry_same_action
    observation.recovery_guidance = assessment.guidance
    return observation


def failure_explanation(observation: "AgentObservation") -> str:
    category = observation.recovery_type or "unknown"
    cause = observation.error or observation.message or "No error detail was returned."
    return f"The {observation.capability} action failed ({category}): {cause}"