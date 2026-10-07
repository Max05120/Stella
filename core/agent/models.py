"""
core/agent/models.py

Shared data models for Stella's agent runtime.

These models deliberately separate:

- the user's goal
- the agent's decision
- capability execution
- observations
- execution history
- overall agent state

The agent should reason about structured state rather than passing
unstructured strings between every layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from uuid import uuid4
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    """
    Overall lifecycle state of an agent run.
    """

    READY = "ready"
    RUNNING = "running"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"  # Legacy compatibility; new runs use specific outcomes.
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    LIMIT_EXCEEDED = "limit_exceeded"


class AgentTerminationReason(str, Enum):
    GOAL_SATISFIED = "goal_satisfied"
    MAX_STEPS = "max_steps"
    RUNTIME_BUDGET = "runtime_budget"
    GLOBAL_STEP_BUDGET = "global_step_budget"
    MODEL_CALL_BUDGET = "model_call_budget"
    MAX_FAILURES = "max_failures"
    MAX_CONSECUTIVE_FAILURES = "max_consecutive_failures"
    NO_PROGRESS = "no_progress"
    REPEATED_DECISION = "repeated_decision"
    REPEATED_FAILED_STRATEGY = "repeated_failed_strategy"
    PLANNER_ERROR = "planner_error"
    INVALID_DECISION = "invalid_decision"
    INVALID_COMPLETION = "invalid_completion"
    INVALID_CONFIRMATION = "invalid_confirmation"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    CONFIRMATION_EXPIRED = "confirmation_expired"
    CONFIRMATION_SUPERSEDED = "confirmation_superseded"
    CONFIRMATION_STILL_BLOCKED = "confirmation_still_blocked"
    USER_CANCELLED = "user_cancelled"
    INTERRUPTED_RUN = "interrupted_run"
    NO_SAFE_ACTION = "no_safe_action"
    UNRECOVERABLE_FAILURE = "unrecoverable_failure"


TERMINAL_STATUSES = frozenset({
    AgentStatus.COMPLETED,
    AgentStatus.BLOCKED,
    AgentStatus.FAILED,
    AgentStatus.CANCELLED,
    AgentStatus.LIMIT_EXCEEDED,
    AgentStatus.ABORTED,
})


class AgentDecisionType(str, Enum):
    """
    The possible things the agent can decide to do next.

    TOOL
        Execute one Stella capability.

    RESPOND
        Say something to the user without ending the task.

    ASK_USER
        More information is required before progress can continue.

    COMPLETE
        The goal has been satisfied.

    ABORT
        Stella cannot safely or reasonably continue.
    """

    TOOL = "tool"
    RESPOND = "respond"
    ASK_USER = "ask_user"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass(slots=True)
class AgentGoal:
    """
    The original goal Stella is trying to accomplish.
    """

    text: str


@dataclass(slots=True)
class AgentDecision:
    """
    A single next-step decision produced by the agent planner.

    Important:
    the agent decides ONE meaningful next action at a time.
    """

    decision_type: AgentDecisionType

    capability: str | None = None
    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    message: str | None = None

    reasoning_summary: str | None = None


@dataclass(slots=True)
class AgentObservation:
    """
    Structured representation of what happened after an action.
    """

    capability: str
    success: bool

    status: str

    data: Any = None
    message: str | None = None
    error: str | None = None

    requires_confirmation: bool = False

    recovery_type: str | None = None

    retry_same_action: bool | None = None

    recovery_guidance: str | None = None


@dataclass(slots=True)
class AgentStep:
    """
    One completed iteration in an agent run.

    A step contains:

        decision -> observation
    """

    number: int
    decision: AgentDecision
    observation: AgentObservation | None = None


@dataclass(frozen=True, slots=True)
class AgentConfirmation:
    confirmation_id: str
    run_id: str
    step_number: int
    capability: str
    arguments_json: str
    created_monotonic: float
    expires_monotonic: float
    explanation: str


@dataclass(slots=True)
class AgentState:
    """
    Complete mutable state for one agent run.

    This becomes the agent's working memory for a task.
    """

    goal: AgentGoal

    status: AgentStatus = AgentStatus.READY

    steps: list[AgentStep] = field(
        default_factory=list
    )

    final_answer: str | None = None

    failure_count: int = 0

    max_steps: int = 15
    max_failures: int = 3

    pending_confirmation: AgentDecision | None = None
    pending_confirmation_step: int | None = None

    run_id: str = field(default_factory=lambda: str(uuid4()))
    termination_reason: AgentTerminationReason | None = None
    final_observation: AgentObservation | None = None

    consecutive_failures: int = 0
    no_progress_steps: int = 0
    productive_steps: int = 0
    max_consecutive_failures: int = 2
    max_no_progress_steps: int = 3
    max_repeated_decisions: int = 3
    decisions_since_progress: dict[str, int] = field(default_factory=dict)
    seen_evidence: set[str] = field(default_factory=set)

    max_runtime_seconds: float = 180.0
    max_global_steps: int = 64
    max_model_calls: int = 24
    decision_calls: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    global_steps_used: int = 0
    started_monotonic: float | None = None
    deadline_monotonic: float | None = None
    finished_monotonic: float | None = None
    runtime_budget_seconds: float | None = None
    elapsed_seconds: float = 0.0
    completion_check_used: bool = False
    conversation_context: list[dict[str, Any]] = field(default_factory=list)
    resume_token: str | None = None
    confirmation_timeout_seconds: float = 60.0
    confirmation: AgentConfirmation | None = None
    confirmation_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        for name in ("max_global_steps", "max_model_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        ttl = self.confirmation_timeout_seconds
        if (isinstance(ttl, bool) or not isinstance(ttl, (int, float))
                or not math.isfinite(ttl) or ttl <= 0):
            raise ValueError("confirmation_timeout_seconds must be finite and positive")
        seconds = self.max_runtime_seconds
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("max_runtime_seconds must be finite and nonnegative")

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def latest_step(self) -> AgentStep | None:
        return self.steps[-1] if self.steps else None

    @property
    def latest_observation(self) -> AgentObservation | None:
        # An incomplete later step must not hide the last actual result.
        return next(
            (step.observation for step in reversed(self.steps)
             if step.observation is not None),
            None,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def can_continue(self) -> bool:
        return (
            not self.is_terminal
            and self.status not in {
                AgentStatus.WAITING_FOR_CONFIRMATION,
                AgentStatus.WAITING_FOR_USER,
            }
            and (
                self.step_count < self.max_steps
                or (
                    self.step_count == self.max_steps
                    and self.latest_step is not None
                    and self.latest_step.observation is not None
                    and self.latest_step.observation.success
                    and not self.completion_check_used
                )
            )
            and self.global_steps_used < self.max_global_steps
            and self.failure_count < self.max_failures
            and self.consecutive_failures < self.max_consecutive_failures
            and self.no_progress_steps < self.max_no_progress_steps
        )

    def inspection_results(self) -> list[dict[str, Any]]:
        """Validated readbacks for supported small snapshots, in step order.

        These are historical observations, never a claim of goal completion.
        Keep the latest successful sample per capability; label its step so a
        later mutation cannot silently make an earlier reading look current.
        Do not expose arbitrary clipboard, file, or UI contents here.
        """
        latest = {}
        for step in self.steps:
            obs = step.observation
            capability = step.decision.capability
            if capability not in {"get_system_volume", "get_frontmost_application"}:
                continue
            latest.pop(capability, None)
            if obs is None or not obs.success or not isinstance(obs.data, dict):
                continue
            if capability == "get_system_volume":
                volume, muted = obs.data.get("volume"), obs.data.get("muted")
                if (type(volume) not in {int, float} or not math.isfinite(volume)
                        or not 0 <= volume <= 100 or type(muted) is not bool):
                    continue
                data = {"volume": volume, "muted": muted}
                summary = f"System volume: {volume:g}%; " + ("muted." if muted else "not muted.")
            else:
                name = obs.data.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                data = {"name": name}
                summary = f"Frontmost application: {name}."
            latest[capability] = {
                "step_number": step.number, "capability": capability,
                "data": data, "summary": summary,
            }
        return sorted(latest.values(), key=lambda row: row["step_number"])

    def termination_diagnostics(self) -> dict[str, Any]:
        observation = (
            self.final_observation
            if self.is_terminal
            else self.latest_observation
        )
        payload = {
            "run_id": self.run_id,
            "status": self.status.value,
            "terminal": self.is_terminal,
            "reason": (
                self.termination_reason.value
                if self.termination_reason is not None else None
            ),
            "summary": self.final_answer,
            "steps": self.step_count,
            "failures": self.failure_count,
            "decision_calls": self.decision_calls,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "global_steps_used": self.global_steps_used,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "runtime_budget_seconds": self.runtime_budget_seconds,
            "runtime_enforcement": "cooperative_boundaries_including_confirmation_wait",
            "completion_check_used": self.completion_check_used,
            "consecutive_failures": self.consecutive_failures,
            "no_progress_steps": self.no_progress_steps,
            "productive_steps": self.productive_steps,
            "distinct_evidence_count": len(self.seen_evidence),
            "decisions_since_progress": dict(self.decisions_since_progress),
            "limits": {
                "max_steps": self.max_steps,
                "confirmation_timeout_seconds": self.confirmation_timeout_seconds,
                "max_runtime_seconds": self.max_runtime_seconds,
                "max_global_steps": self.max_global_steps,
                "max_model_calls": self.max_model_calls,
                "max_failures": self.max_failures,
                "max_consecutive_failures": self.max_consecutive_failures,
                "max_no_progress_steps": self.max_no_progress_steps,
                "max_repeated_decisions": self.max_repeated_decisions,
            },
            "recovery": {
                "planner_failure": self.termination_reason == AgentTerminationReason.PLANNER_ERROR,
                "failed_steps": [
                    {
                        "step_number": step.number,
                        "capability": step.decision.capability,
                        "category": step.observation.recovery_type or "unknown",
                        "retry_same_action": step.observation.retry_same_action is True,
                        "guidance": step.observation.recovery_guidance,
                        "error": step.observation.error,
                    }
                    for step in self.steps
                    if step.observation is not None and not step.observation.success
                    and not (step.observation.status == "blocked" and step.observation.requires_confirmation)
                ],
            },
            "successful_steps": sum(
                bool(step.observation and step.observation.success)
                for step in self.steps
            ),
            "inspection_results": self.inspection_results(),
            "step_trace": [
                {"step_number": step.number,
                 "capability": step.decision.capability,
                 "success": step.observation.success if step.observation else None,
                 "status": step.observation.status if step.observation else "unobserved",
                 "recovery_type": step.observation.recovery_type if step.observation else None}
                for step in self.steps
            ],
            "pending_confirmation_step": self.pending_confirmation_step,
            "confirmation": asdict(self.confirmation) if self.confirmation else None,
            "confirmation_events": self.confirmation_events,
            "final_observation": (
                asdict(observation) if observation is not None else None
            ),
        }
        # Capabilities can return Path and other non-JSON-native values.
        return json.loads(json.dumps(payload, default=str, ensure_ascii=False))