"""Run-wide budgets; synchronous calls are checked, never forcibly killed."""
from __future__ import annotations

from time import monotonic

from core.agent.models import AgentState, AgentTerminationReason


class AgentBudgetExceeded(Exception):
    def __init__(self, reason: AgentTerminationReason, message: str):
        super().__init__(message)
        self.reason = reason


def touch_runtime(state: AgentState) -> None:
    if state.started_monotonic is not None and state.finished_monotonic is None:
        state.elapsed_seconds = max(0.0, monotonic() - state.started_monotonic)


def start_runtime(state: AgentState) -> None:
    if state.started_monotonic is None:
        state.started_monotonic = monotonic()
        state.runtime_budget_seconds = state.max_runtime_seconds
        state.deadline_monotonic = state.started_monotonic + state.max_runtime_seconds


def check_runtime(state: AgentState) -> None:
    start_runtime(state)
    touch_runtime(state)
    if monotonic() >= state.deadline_monotonic:
        raise AgentBudgetExceeded(
            AgentTerminationReason.RUNTIME_BUDGET,
            "I stopped because this task reached its elapsed-time budget. "
            "Any action already in flight was allowed to return; its result is preserved.",
        )


def freeze_runtime(state: AgentState) -> None:
    if state.started_monotonic is not None and state.finished_monotonic is None:
        state.finished_monotonic = monotonic()
        state.elapsed_seconds = max(0.0, state.finished_monotonic - state.started_monotonic)


def claim_operation(state: AgentState, kind: str) -> None:
    """Reserve BEFORE invoking work. Attempts count even if the call fails.

One global unit is one planner decision cycle, one model attempt (including
repair), or one capability-adapter dispatch (including confirmed dispatch).
"""
    counters = {"decision": "decision_calls", "model": "model_calls", "tool": "tool_calls"}
    if kind not in counters:
        raise ValueError(f"Unknown budget operation: {kind}")
    check_runtime(state)
    if state.global_steps_used >= state.max_global_steps:
        raise AgentBudgetExceeded(
            AgentTerminationReason.GLOBAL_STEP_BUDGET,
            "I stopped because this task reached its global operation budget.",
        )
    if kind == "model" and state.model_calls >= state.max_model_calls:
        raise AgentBudgetExceeded(
            AgentTerminationReason.MODEL_CALL_BUDGET,
            "I stopped because this task reached its model-attempt budget, including repairs.",
        )
    state.global_steps_used += 1
    name = counters[kind]
    setattr(state, name, getattr(state, name) + 1)


def budget_summary(state: AgentState) -> str:
    remaining_tools = max(0, state.max_steps - state.step_count)
    return (
        f"Tool steps remaining: {remaining_tools}\n"
        f"Model attempts used: {state.model_calls}/{state.max_model_calls}\n"
        f"Global operations used: {state.global_steps_used}/{state.max_global_steps}\n"
        + (
            "NO TOOL STEPS REMAIN. Choose COMPLETE only if observations support "
            "the entire goal; otherwise ABORT. Do not propose another tool."
            if remaining_tools == 0 else
            "Use the remaining budget only for unfinished requirements."
        )
    )