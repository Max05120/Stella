"""
Phase 6G.2 diagnostic.

Tests whether Stella changes strategy after a real,
classified capability failure.
"""

from datetime import datetime
from pathlib import Path

from core.agent import (
    AgentGoal,
    AgentLoop,
    AgentState,
)


def separator(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_state(state: AgentState):

    print("STATUS:", state.status.value)
    print("ANSWER:", state.final_answer)
    print("STEPS:", state.step_count)
    print("FAILURES:", state.failure_count)

    for step in state.steps:

        print()
        print(f"Step {step.number}")
        print(
            " Capability:",
            step.decision.capability,
        )
        print(
            " Arguments:",
            step.decision.arguments,
        )

        observation = step.observation

        if observation is None:
            print(" Observation: None")
            continue

        print(
            " Success:",
            observation.success,
        )
        print(
            " Status:",
            observation.status,
        )
        print(
            " Message:",
            observation.message,
        )
        print(
            " Error:",
            observation.error,
        )
        print(
            " Recovery type:",
            observation.recovery_type,
        )
        print(
            " Retry same action:",
            observation.retry_same_action,
        )
        print(
            " Recovery guidance:",
            observation.recovery_guidance,
        )


def main():

    separator(
        "STELLA PHASE 6G.2 — LIVE RECOVERY"
    )

    timestamp = (
        datetime.now()
        .strftime("%H%M%S")
    )

    folder_name = (
        f"Stella Recovery Test {timestamp}"
    )

    folder_path = (
        Path.home()
        / "Downloads"
        / folder_name
    )

    # Deliberately create it BEFORE Stella runs.
    folder_path.mkdir(
        parents=False,
        exist_ok=False,
    )

    print(
        "Pre-created folder:",
        folder_path,
    )

    loop = AgentLoop()

    state = AgentState(
        goal=AgentGoal(
            text=(
                "Create a folder named "
                f"'{folder_name}' "
                "in my Downloads folder, "
                "then open that folder."
            )
        ),
        max_steps=8,
        max_failures=3,
    )

    separator(
        "RUNNING AGENT"
    )

    state = loop.run(
        state
    )

    separator(
        "FINAL STATE"
    )

    print_state(
        state
    )

    separator(
        "FILESYSTEM CHECK"
    )

    print(
        "Folder exists:",
        folder_path.exists(),
    )

    separator(
        "DONE"
    )


if __name__ == "__main__":
    main()