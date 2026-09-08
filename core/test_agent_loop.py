"""
Diagnostic for Stella's autonomous agent loop.

WARNING:

Unlike test_agent_planner.py, this diagnostic ACTUALLY EXECUTES safe
macOS capabilities.

Test 1 Launches Spotify.

Test 2 creates and opens a uniquely named folder in Downloads.
"""

from datetime import datetime
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


def print_result(state: AgentState):
    print()
    print("FINAL STATUS:", state.status.value)
    print("FINAL ANSWER:", state.final_answer)
    print("STEPS:", state.step_count)

    print("FAILURES:", state.failure_count)

    for step in state.steps:
        print()
        print(f"Step {step.number}")
        print(" Decision:", step.decision.decision_type.value)

        print(" Capability:", step.decision.capability)
        print("Arguments:", step.decision.arguments)

        if step.observation is None:

            print("Observation: None")

            continue
        print(" Success:", step.observation.success)
        print("Status:", step.observation.status)
        print(" Data:", step.observation.data)
        print(" Message:", step.observation.message)
        print(" Error:", step.observation.error)

def run_goal(
        loop: AgentLoop,
        goal: str,
):
    separator(f"GOAL: {goal}")

    state = AgentState(
        goal=AgentGoal(
            text=goal
        ),
        max_steps=8,
        max_failures=3,
    )

    result = loop.run(state)

    print_result(result)

def main():

    separator(
        "STELLA AUTONOMOUS AGENT LOOP"
    )

    loop = AgentLoop()

    print("Agent loop intialized.")

    #--------------------------------------------------------------------------------
    # Test 1
    #
    # Real multi-step task:
    # Open Spotify -> observe success -> inspect frontmost application -> complete
    # -------------------------------------------------------------------------------

    run_goal(
        loop,
        (
            "Open Spotify and then tell me "
            "what application is currently frontmost."     
        ),
    )

    #-------------------------------------------------------------------
    # TEST 2
    #
    # Unique folder name prevents FileExistsError
    # when running this multiple times
    #--------------------------------------------------------------------

    timestamp = datetime.now().strftime("%H%M%S")

    folder_name = f"Stella Agent Test {timestamp}" 

    run_goal(
        loop,
        (
            f"Create a folder called "
            f"'{folder_name}' "
            f"in my Downloads folder and then open it."
        ),
    )

    separator("DONE")


if __name__ == "__main__":
    main()