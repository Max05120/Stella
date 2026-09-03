"""
models.py

Shared types for Stella's Mac capability system.

A capability describes something Stella is actually able to execute.
It is deliberately separate from mac_knowledge, which describes what
macOS can theoretically do.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class CapabilityRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    IMPLEMENTABLE = "implementable"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"


@dataclass
class Capability:
    name: str
    family: str
    description: str

    handler: Callable[..., Any]

    action: str | None = None
    
    risk: CapabilityRisk = CapabilityRisk.LOW

    permissions: list[str] = field(
        default_factory=list
    )

    requires_confirmation: bool = False
    supports_undo: bool = False

    implementation: str | None = None

    intents: list[str] = field(
        default_factory=list
    )


@dataclass
class ExecutionResult:
    capability: str
    status: ExecutionStatus

    message: str

    data: Any = None
    error: str | None = None