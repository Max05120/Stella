"""
Diagnostic for Stella Agent Planner.

IMPORTANT:

This test asks the agent to DECIDE what it would do.

It does NOT execute the chosen capability.
"""

from core.agent import (
    AgentGoal,
    AgentPlanner,
    AgentPlannerError,
    AgentState,
)

def separator(
        title: str,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

def test_goal(
        planner: AgentPlanner,
        goal_text: str,
):
    separator(
        f"GOAL: {goal_text}"
    )

    state = AgentState(
        goal =AgentGoal(
            text=goal_text
        )
    )

    try:
        decision = planner.decide(
            state
        )
    
    except AgentPlannerError as exc:
        print(
            "PLANNER ERROR:",
            exc,
        )

        return
    print(
        "Decision type:",
        decision.decision_type.value,
    )

    print(
        "Capability:",
        decision.capability,
    )

    print(
        "Arguments:",
        decision.arguments,
    )
    print(
        "Message:",
        decision.message,
    )
    print(
        "Reasoning summary: ",
        decision.reasoning_summary,
    )

def main():

    separator(
        "STELLA AGENT PLANNER"
    )

    planner = AgentPlanner()

    print(
        "Planner initiated."
    )

    #---------------------------------------------
    # Test 1
    # Should probably inspect system volume.
    #---------------------------------------------

    test_goal(
        planner,
        (
            "Find out what my current "
            "system volume is."
        ),
    )
    # ---------------------------------------------------------
    # Test 2
    #
    # Should choose an application capability.
    # ---------------------------------------------------------

    test_goal(
        planner,
        "Open Spotify.",
    )

    # ---------------------------------------------------------
    # Test 3
    #
    # Multi-step goal.
    #
    # The important thing:
    # planner should choose ONE sensible first step.
    # ---------------------------------------------------------

    test_goal(
        planner,
        (
            "Open Spotify and then tell me "
            "what application is currently frontmost."
        ),
    )
    # ---------------------------------------------------------
    # Test 4
    #
    # Another genuinely multi-step goal.
    # ---------------------------------------------------------

    test_goal(
        planner,
        (
            "Create a folder called Agent Test "
            "in my Downloads folder and then open it."
        ),
    )

    separator(
        "DONE — NOTHING WAS EXECUTED"
    )

if __name__ == "__main__":
    main()