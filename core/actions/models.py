"""
core/actions/models.py

Structured action representations used between Stella's natural-language
input and the capability system.

The important rule here is:

    natural language
        ↓
    ActionRequest
        ↓
    capability decision / execution

The planner may change in the future, but ActionRequest remains the
structured boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ActionRequest:
    """
    A normalized representation of a user-requested action.
    """

    raw_text: str

    # Semantic action being requested:
    # open, reveal, etc.
    action: str

    # What kind of thing is being acted upon:
    # application, url, path
    target_type: str

    # Human-readable normalized target.
    target: str

    # Arguments expected by the eventual capability handler.
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.raw_text = self.raw_text.strip()
        self.action = self.action.strip().lower()
        self.target_type = self.target_type.strip().lower()
        self.target = self.target.strip()

        if not self.raw_text:
            raise ValueError("ActionRequest.raw_text cannot be empty")

        if not self.action:
            raise ValueError("ActionRequest.action cannot be empty")

        if not self.target_type:
            raise ValueError("ActionRequest.target_type cannot be empty")

        if not self.target:
            raise ValueError("ActionRequest.target cannot be empty")


@dataclass(slots=True)
class PlanningResult:
    """
    Result returned by the deterministic action planner.

    Keeping planning failure separate from ActionRequest means downstream
    code never has to guess whether a malformed request is executable.
    """

    understood: bool
    request: ActionRequest | None = None
    reason: str | None = None

    @classmethod
    def success(cls, request: ActionRequest) -> "PlanningResult":
        return cls(
            understood=True,
            request=request,
            reason=None,
        )

    @classmethod
    def failure(cls, reason: str) -> "PlanningResult":
        return cls(
            understood=False,
            request=None,
            reason=reason,
        )