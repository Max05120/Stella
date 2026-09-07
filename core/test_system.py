from core.actions.planner import plan_action
from core.actions.runner import run_action
from core.capabilities import initialize_capabilities
from core.capabilities.registry import registry
from core.capabilities.executor import execute_capability


def divider(title: str):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def test_registration():

    divider("1. SYSTEM CAPABILITY REGISTRATION")

    initialize_capabilities()

    required = [
        "get_frontmost_application",
        "get_clipboard",
        "set_clipboard",
        "get_system_volume",
        "set_system_volume",
        "mute_system_volume",
        "unmute_system_volume",
    ]

    names = registry.names()

    for name in required:
        print(
            "✓" if name in names else "✗",
            name,
        )


def test_direct():

    divider("2. DIRECT CAPABILITY EXECUTION")

    for capability in [
        "get_frontmost_application",
        "get_clipboard",
        "get_system_volume",
    ]:

        result = execute_capability(
            capability,
            {},
        )

        print(
            f"\n{capability}"
        )

        print(
            " status:",
            result.status,
        )

        print(
            " data:",
            result.data,
        )

        print(
            " error:",
            result.error,
        )


def test_pipeline():

    divider("3. FULL ACTION PIPELINE")

    queries = [
        # "What's the current app?",
        # "What's in my clipboard?",
        # "What's the current volume?",
        "Set the volume to 35%",
        "Mute the sound",
        "Unmute the sound",
        "Copy Stella system test to my clipboard",
    ]

    for query in queries:

        print(
            f"\nQUERY: {query}"
        )

        result = run_action(
            query
        )

        print(
            " status:",
            result.status,
        )

        print(
            " capability:",
            result.capability,
        )

        if result.request:
            print(
                " request:",
                result.request.action,
                "/",
                result.request.target_type,
            )

        if result.execution:
            print(
                " data:",
                result.execution.data,
            )

            print(
                " error:",
                result.execution.error,
            )


if __name__ == "__main__":

    # test_registration()
    # test_direct()
    test_pipeline()