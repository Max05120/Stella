"""Shared, deterministic progress policy for planner and runtime.

This detects exact-call/evidence loops. It does not prove semantic progress
toward an arbitrary natural-language goal.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from core.agent.models import (
    AgentDecision, AgentDecisionType, AgentObservation, AgentState,
    AgentTerminationReason,
)


from core.agent.recovery import READ_ONLY_CAPABILITIES

@dataclass(frozen=True, slots=True)
class ProgressViolation:
    reason: AgentTerminationReason
    message: str


def fingerprint(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def action_key(decision: AgentDecision) -> str:
    # Message/reasoning changes and dict-key order cannot disguise a repeat.
    return fingerprint([decision.capability, decision.arguments])


def confirmation_gate(observation: AgentObservation) -> bool:
    return (
        not observation.success and observation.status == "blocked"
        and observation.requires_confirmation
    )


def check_action(state: AgentState, decision: AgentDecision) -> ProgressViolation | None:
    """Inspect without modifying state. Safe to call during planner repair."""
    if decision.decision_type != AgentDecisionType.TOOL or not decision.capability:
        return None

    key = action_key(decision)
    read_only = decision.capability in READ_ONLY_CAPABILITIES
    matches = [
        step for step in state.steps
        if step.decision.decision_type == AgentDecisionType.TOOL
        and action_key(step.decision) == key
        and step.observation is not None
        and not confirmation_gate(step.observation)
    ]

    if not read_only and any(step.observation.success for step in matches):
        return ProgressViolation(
            AgentTerminationReason.REPEATED_DECISION,
            "This exact state-changing action already succeeded in this run. "
            "Use that result and continue with an unfinished requirement.",
        )

    failures = [step.observation for step in matches if not step.observation.success]
    if failures and (
        not read_only
        or failures[-1].retry_same_action is not True
        or len(failures) >= 2
    ):
        return ProgressViolation(
            AgentTerminationReason.REPEATED_FAILED_STRATEGY,
            "This exact failed strategy cannot be retried again. Only one "
            "retry of an explicitly retryable read-only call is allowed. "
            "Choose a different safe strategy, ask the user, or stop.",
        )

    recovery_violation = check_recovery_action(state, decision)
    if recovery_violation is not None:
        return recovery_violation

    if state.decisions_since_progress.get(key, 0) >= state.max_repeated_decisions:
        return ProgressViolation(
            AgentTerminationReason.REPEATED_DECISION,
            "This tool decision has reached its repetition limit without "
            "new evidence or a successful new action.",
        )
    return None


def check_recovery_action(state: AgentState, decision: AgentDecision) -> ProgressViolation | None:
    """Recovery-only guard; safe for an already-reserved confirmation step."""
    read_only = decision.capability in READ_ONLY_CAPABILITIES
    # A different name/argument cannot make an uncertain mutation safe.
    if not read_only:
        for step in state.steps:
            obs = step.observation
            if obs is None or obs.success or confirmation_gate(obs):
                continue
            denied = obs.recovery_type == "permission_denied"
            uncertain = (
                step.decision.capability not in READ_ONLY_CAPABILITIES
                and obs.recovery_type in {"timeout", "transient", "unknown"}
            )
            if denied or uncertain:
                return ProgressViolation(
                    AgentTerminationReason.NO_SAFE_ACTION,
                    "An earlier action was denied permission or has an uncertain "
                    "outcome. Only inspection is allowed during recovery in this "
                    "run. Ask the user or stop before another state-changing action.",
                )
    return None


def record_dispatch(state: AgentState, decision: AgentDecision) -> None:
    """Count a new tool step once; approval continues the already-counted step."""
    key = action_key(decision)
    state.decisions_since_progress[key] = state.decisions_since_progress.get(key, 0) + 1


def record_result(state: AgentState, decision: AgentDecision, observation: AgentObservation) -> None:
    """Called once per resolved tool step, including the result after approval.

Initial confirmation gates call neither this function nor a failure counter.
An action remaining blocked AFTER approval is a real resolved failure.
"""
    if not observation.success:
        state.failure_count += 1
        state.consecutive_failures += 1
        state.no_progress_steps += 1
        return

    state.consecutive_failures = 0
    if decision.capability in READ_ONLY_CAPABILITIES:
        # Empty lists/dicts are valid evidence; only None uses message fallback.
        evidence = (
            ["data", observation.data]
            if observation.data is not None
            else ["message", observation.message]
        )
        key = fingerprint([action_key(decision), evidence])
        productive = key not in state.seen_evidence
        state.seen_evidence.add(key)
    else:
        # A successful new mutation is a progress proxy, not proof of the goal.
        productive = True

    if productive:
        state.productive_steps += 1
        state.no_progress_steps = 0
        state.decisions_since_progress.clear()
    else:
        state.no_progress_steps += 1