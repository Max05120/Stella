"""
Diagnostic for Stella Phase 6F.

Tests:

1. Agent reaches a destructive action.
2. Executor blocks it for confirmation.
3. AgentState preserves the exact pending decision.
4. Confirmation resumes the SAME run.
5. The confirmed tool executes.
6. Agent continues toward completion.
"""

from datetime import datetime
from pathlib import Path

from core.agent import (
    AgentGoal,
    AgentLoop,
    AgentState,
    AgentStatus,
)


def separator(
    title: str,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_state(
    state: AgentState,
):

    print(
        "STATUS:",
        state.status.value,
    )

    print(
        "ANSWER:",
        state.final_answer,
    )

    print(
        "STEPS:",
        state.step_count,
    )

    print(
        "FAILURES:",
        state.failure_count,
    )

    print(
        "PENDING:",
        state.pending_confirmation,
    )

    print(
        "PENDING STEP:",
        state.pending_confirmation_step,
    )

    for step in state.steps:

        print()
        print(
            f"Step {step.number}"
        )

        print(
            " Capability:",
            step.decision.capability,
        )

        print(
            " Arguments:",
            step.decision.arguments,
        )

        if step.observation is None:

            print(
                " Observation: None"
            )

            continue

        print(
            " Success:",
            step.observation.success,
        )

        print(
            " Status:",
            step.observation.status,
        )

        print(
            " Message:",
            step.observation.message,
        )

        print(
            " Error:",
            step.observation.error,
        )


def main():

    separator(
        "STELLA PHASE 6F — CONFIRMATION RESUME"
    )

    loop = AgentLoop()

    # ---------------------------------------------------------
    # Create harmless disposable test file.
    # ---------------------------------------------------------

    timestamp = (
        datetime.now()
        .strftime("%H%M%S")
    )

    test_path = Path.home() / (
        "Downloads"
    ) / (
        f"Stella Confirmation Test {timestamp}.txt"
    )

    test_path.write_text(
        "Disposable Stella agent confirmation test.",
        encoding="utf-8",
    )

    print(
        "Created test file:",
        test_path,
    )

    # ---------------------------------------------------------
    # Start agent goal.
    # ---------------------------------------------------------

    state = AgentState(
        goal=AgentGoal(
            text=(
                "Move this file to Trash: "
                f"{test_path}"
            )
        ),
        max_steps=8,
        max_failures=3,
    )

    separator(
        "FIRST RUN"
    )

    state = loop.run(
        state
    )

    print_state(
        state
    )

    # ---------------------------------------------------------
    # Must now be paused.
    # ---------------------------------------------------------

    if (
        state.status
        != AgentStatus.WAITING_FOR_CONFIRMATION
    ):
        print()
        print(
            "ERROR: Agent did not pause "
            "for confirmation."
        )

        return

    separator(
        "APPROVING PENDING ACTION"
    )

    state = loop.confirm(
        state,
        approved=False,
    )

    print_state(
        state
    )

    separator(
        "FILE CHECK"
    )

    print(
        "Original file still exists:",
        test_path.exists(),
    )

    separator(
        "DONE"
    )


if __name__ == "__main__":
    main()