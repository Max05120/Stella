"""
decision.py

High-level capability decision layer.

AVAILABLE
    Stella has a matching executable capability.

IMPLEMENTABLE
    No tool exists, but mac_knowledge indicates the action
    can likely be implemented.

UNKNOWN
    No tool exists and Stella lacks sufficient implementation
    evidence.
"""

from dataclasses import dataclass

from core.capabilities.models import (
    Capability,
)

from core.capabilities.resolver import (
    resolve_best_capability,
)

from core.capabilities.gap_analyzer import (
    CapabilityGap,
    analyze_capability_gap,
)


@dataclass
class CapabilityDecision:
    status: str

    capability: Capability | None = None
    score: float | None = None

    gap: CapabilityGap | None = None


def decide_capability(
    query: str,
) -> CapabilityDecision:

    match = resolve_best_capability(
        query
    )

    if match is not None:

        return CapabilityDecision(
            status="available",
            capability=match.capability,
            score=match.score,
        )

    gap = analyze_capability_gap(
        query
    )

    return CapabilityDecision(
        status=gap.status,
        gap=gap,
    )