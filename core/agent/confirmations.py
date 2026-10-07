"""In-memory confirmation snapshots and audit events (Phase 6F.1)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from math import ceil
from uuid import uuid4

from core.agent import budgets
from core.agent.models import AgentConfirmation, AgentDecision, AgentState


def canonical_arguments(arguments: dict) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)


def audit_confirmation(state: AgentState, event: str, record: AgentConfirmation, **details) -> None:
    entry = {
        "event": event,
        "run_id": record.run_id,
        "confirmation_id": record.confirmation_id,
        "step_number": record.step_number,
        "capability": record.capability,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    state.confirmation_events.append(entry)
    print("[AGENT CONFIRMATION] " + json.dumps(entry, ensure_ascii=False))


def create_confirmation(state: AgentState, decision: AgentDecision, step_number: int) -> AgentConfirmation:
    now = budgets.monotonic()
    expires = min(now + state.confirmation_timeout_seconds, state.deadline_monotonic)
    seconds = max(0, ceil(expires - now))
    arguments_json = canonical_arguments(decision.arguments)
    action = decision.capability.replace("_", " ")
    explanation = (
        f"I'm about to {action}. "
        + (f"Arguments: {arguments_json}. " if decision.arguments else "")
        + f"Reply yes to approve, no to reject, or cancel to stop the task. "
        f"This approval is valid for up to {seconds} seconds."
    )
    record = AgentConfirmation(
        confirmation_id=str(uuid4()),
        run_id=state.run_id,
        step_number=step_number,
        capability=decision.capability,
        arguments_json=arguments_json,
        created_monotonic=now,
        expires_monotonic=expires,
        explanation=explanation,
    )
    state.confirmation = record
    audit_confirmation(state, "requested", record, valid_for_seconds=round(expires - now, 3))
    return record


def record_matches(state: AgentState, decision: AgentDecision, step_number: int) -> bool:
    record = state.confirmation
    if record is None:
        return False
    try:
        return (
            record.run_id == state.run_id
            and record.step_number == step_number
            and record.capability == decision.capability
            and record.arguments_json == canonical_arguments(decision.arguments)
        )
    except (TypeError, ValueError):
        return False


def confirmation_expired(state: AgentState) -> bool:
    return state.confirmation is None or budgets.monotonic() >= state.confirmation.expires_monotonic


def close_confirmation(state: AgentState, reason) -> None:
    record = state.confirmation
    if record is not None:
        event = {
            "confirmation_rejected": "rejected",
            "user_cancelled": "cancelled",
            "confirmation_expired": "expired",
        }.get(reason.value, "invalidated")
        audit_confirmation(state, event, record, reason=reason.value)
    state.confirmation = None