"""
Phase 6F.2 diagnostic.

Tests autonomous agent confirmation persistence
through StellaEngine.
"""

from datetime import datetime
from pathlib import Path

from core.engine import StellaEngine


def separator(
    title: str,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    engine = StellaEngine()

    conversation_id = (
        engine.create_conversation()
    )

    timestamp = (
        datetime.now()
        .strftime("%H%M%S")
    )

    test_path = (
        Path.home()
        / "Downloads"
        / (
            "Stella Engine Confirmation "
            f"{timestamp}.txt"
        )
    )

    test_path.write_text(
        (
            "Disposable Phase 6F.2 "
            "confirmation test."
        ),
        encoding="utf-8",
    )

    print(
        "Created:",
        test_path
    )

    # -------------------------------------------------
    # 1. Start autonomous agent goal.
    # -------------------------------------------------

    separator(
        "START AGENT GOAL"
    )

    response = (
        engine.run_agent_goal(
            (
                "Move this file to Trash: "
                f"{test_path}"
            ),
            conversation_id,
        )
    )

    print(
        "Stella:",
        response["answer"],
    )

    print(
        "Tools:",
        response["tools_used"],
    )

    print(
        "File exists:",
        test_path.exists(),
    )

    print(
        "Active agent stored:",
        (
            conversation_id
            in engine._active_agent_states
        ),
    )

    # -------------------------------------------------
    # 2. Confirm through NORMAL chat().
    # -------------------------------------------------

    separator(
        "USER SAYS YES/NO THROUGH CHAT"
    )

    response = engine.chat(
        "yes",
        conversation_id,
    )

    print(
        "Stella:",
        response["answer"],
    )

    print(
        "Tools:",
        response["tools_used"],
    )

    print(
        "File exists:",
        test_path.exists(),
    )

    print(
        "Active agent stored:",
        (
            conversation_id
            in engine._active_agent_states
        ),
    )

    separator(
        "DONE"
    )


if __name__ == "__main__":
    main()