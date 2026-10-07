"""
core/agent/observer.py

Converts Stella capability execution results into observations
that the agent runtime can reason about.
"""

from __future__ import annotations

from typing import Any
from core.agent.recovery import annotate_failure
from core.agent.models import AgentObservation
from core.capabilities.models import (
    ExecutionResult,
    ExecutionStatus,
)


def observe_execution(
    execution: ExecutionResult,
    *,
    requires_confirmation: bool = False,
) -> AgentObservation:
    """
    Convert a capability ExecutionResult into an AgentObservation.

    The capability executor belongs to the action world.

    AgentObservation belongs to the agent world.

    Keeping the two separate prevents the agent runtime from becoming
    tightly coupled to low-level capability implementation details.
    """

    status = execution.status

    status_value = getattr(
        status,
        "value",
        status,
    )

    success = (
        status
        == ExecutionStatus.SUCCESS
        or str(status_value).lower()
        == "success"
    )

    observation = AgentObservation(
        capability=execution.capability,
        success=success,
        status=str(status_value),
        data=execution.data,
        message=execution.message,
        error=execution.error,
        requires_confirmation=requires_confirmation,
    )

    return annotate_failure(observation)