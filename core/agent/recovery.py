"""
core/agetn/recovery.py

Structured failure classification and recovery guidance
for Stella's autonomous agent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.models import (
        AgentObservation,
    )

class RecoveryType(str, Enum):

    ALREADY_EXISTS = "already_exists"

    NOT_FOUND = "not_found" 

    PERMISSION_DENIED = "permission_denied"

    INVALID_ARGUMENT = "invalid_argument"

    CONFIRMATION_REQUIRED = "confirmation_required"

    TRANSIENT = "transient" 

    UNKNOWN = "unknown"


@dataclass(slots=True)
class RecoveryAssessment:

    recovery_type: RecoveryType

    retry_same_action: bool

    guidance: str


def assess_failure(
        observation: "AgentObservation",
) -> RecoveryAssessment:
    """
    Classify a failed observation and return deterministic
    recovery guidance for the planner.
    """

    #------------------------------------------------------------
    # Confirmation is not a normal execution failure
    #------------------------------------------------------------

    if (
        observation.requires_confirmation
        and observation.status == "blocked"
    ):
        return RecoveryAssessment(
            recovery_type=(
                RecoveryType.CONFIRMATION_REQUIRED
            ),
            retry_same_action=False,
            guidance=(
                "The action requires user confirmation. " \
                "Do not retry or replace it automatically."
            ),
        )
    
    text = " ".join(
        part 
        for part in (
            observation.error,
            observation.message,
        )
        if part
    ).lower()

    #------------------------------------------------------------
    # Already exists
    #------------------------------------------------------------

    if (
        "already exists" in text
        or "file exists" in text
    ):
        return RecoveryAssessment(
            recovery_type=(
                RecoveryType.ALREADY_EXISTS
            ),
            retry_same_action=False,
            guidance=(
                "The requested resource already exist. " \
                "Do not repeat the same creation action. " \
                "Use the existing resource if that still " \
                "satisfies the goal, inspect it if needed, " \
                "or choose a differernt valid strategy."
            ),
        )
    
    #------------------------------------------------------------
    # Not found
    #------------------------------------------------------------

    if (
        "not found" in text or
        "no such file" in text or
        "does not exist" in text or
        "could not find" in text
    ):
        return RecoveryAssessment(
            recovery_type=(
                RecoveryType.NOT_FOUND
            ),
            retry_same_action=False,
            guidance=(
                "The required resource could not be found. " \
                "Do not blindly repeat the same action. " \
                "Use an available discovery or inspection " \
                "capability to locate the correct resource, " \
                "or ask the user if it cannot be resolved."
            ),
        )
    
    #------------------------------------------------------------
    # Permission failure
    #------------------------------------------------------------

    if (
        "permission denied" in text or
        "not permitted" in text or
        "operation not permitted" in text or
        "access denied" in text
    ):
        return RecoveryAssessment(
            recovery_type = RecoveryType.PERMISSION_DENIED,
            retry_same_action=False,
            guidance=(
                "The action was denied by system permissions. " \
                "Repeating the same action is unlikely to help. " \
                "Choose another valid strategy if one exists, " \
                "otherwise explain the permission problem " \
                "to the user."
            ),
        )
    
    #------------------------------------------------------------
    # Invalid arguments / Malformed request
    #------------------------------------------------------------

    if (
        "invalid argument" in text or
        "unexpected keyword" in text or
        (
            "missing" in text and 
            "argument" in text
        ) 
        or "required positional" in text
    ):
        
        return RecoveryAssessment(
            recovery_type= RecoveryType.INVALID_ARGUMENT,
            retry_same_action=False,
            guidance=(
                "The tool invocation used invalid arguments. "
                "Do not repeat the identical call. "
                "Correct the arguments using the capability "
                "schema before trying again."
            ),
        )
    
    #------------------------------------------------------------
    # Potentially transient
    #------------------------------------------------------------

    if (
        "temporarily unavailable" in text or
        "timed out" in text or
        "timeout" in text or 
        "try again" in text
    ):
        
        return RecoveryAssessment(
            recovery_type=RecoveryType.TRANSIENT,
            retry_same_action=True,
            guidance=(
                "The failure may be temporary. " \
                "One retry of the same action is permitted " \
                "if it still advances the goal."
            ),
        )
    
    #------------------------------------------------------------
    # Unknown
    #------------------------------------------------------------

    return RecoveryAssessment(
        recovery_type=RecoveryType.UNKNOWN,
        retry_same_action=False,
        guidance=(
            "The failure could not be classified confidently. " \
            "Do not blindly repeat the identical failed action. " \
            "Inspect available observations, choose a differernt " \
            "strategy, ask the user if necessary, or abort " \
            "fi no safe path remains."
        ),
    )