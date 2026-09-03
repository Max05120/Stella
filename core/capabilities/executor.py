"""
executor.py

Controlled execution boundary for Stella's capabilities.

The planner should never call capability handlers directly.
Everything passes through this executor.
"""

import inspect

from core.capabilities.models import (
    ExecutionResult,
    ExecutionStatus,
)

from core.capabilities.registry import (
    registry,
)


def execute_capability(
    name: str,
    arguments: dict | None = None,
    confirmed: bool = False,
) -> ExecutionResult:

    arguments = arguments or {}

    capability = registry.get(name)

    if capability is None:
        return ExecutionResult(
            capability=name,
            status=ExecutionStatus.NOT_FOUND,
            message=(
                f"Capability '{name}' "
                f"is not currently available."
            ),
        )

    if (
        capability.requires_confirmation
        and not confirmed
    ):
        return ExecutionResult(
            capability=name,
            status=ExecutionStatus.BLOCKED,
            message=(
                f"Capability '{name}' requires "
                f"user confirmation before execution."
            ),
        )

    try:
        signature = inspect.signature(
            capability.handler
        )

        signature.bind(**arguments)

    except TypeError as exc:
        return ExecutionResult(
            capability=name,
            status=ExecutionStatus.FAILED,
            message=(
                "Invalid capability arguments."
            ),
            error=str(exc),
        )

    try:
        result = capability.handler(
            **arguments
        )

        return ExecutionResult(
            capability=name,
            status=ExecutionStatus.SUCCESS,
            message=(
                f"Capability '{name}' "
                f"executed successfully."
            ),
            data=result,
        )

    except Exception as exc:
        return ExecutionResult(
            capability=name,
            status=ExecutionStatus.FAILED,
            message=(
                f"Capability '{name}' failed."
            ),
            error=str(exc),
        )