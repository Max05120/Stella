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

from dataclasses import dataclass, field
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
    ABORTED = "aborted"


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

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def latest_step(self) -> AgentStep | None:
        if not self.steps:
            return None

        return self.steps[-1]

    @property
    def latest_observation(
        self,
    ) -> AgentObservation | None:

        step = self.latest_step

        if step is None:
            return None

        return step.observation

    def can_continue(self) -> bool:
        """
        Hard safety guard for the future agent loop.
        """

        if self.status in {
            AgentStatus.WAITING_FOR_CONFIRMATION,
            AgentStatus.WAITING_FOR_USER,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.ABORTED,
        }:
            return False

        if self.step_count >= self.max_steps:
            return False

        if self.failure_count >= self.max_failures:
            return False

        return True