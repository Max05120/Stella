from core.actions.runner import run_action
from core.capabilities import initialize_capabilities
from core.capabilities.executor import execute_capability
from core.capabilities.registry import registry


def divider(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def test_registration():

    divider("1. REGISTRATION")

    initialize_capabilities()

    required = [
        "get_running_applications",
        "activate_application",
        "hide_application",
        "quit_application",
        "get_application_windows",
        "get_frontmost_window",
        "focus_window",
        "move_window",
        "resize_window",
    ]

    names = registry.names()

    for name in required:
        print(
            "✓" if name in names else "✗",
            name,
        )


def test_direct():

    divider("2. DIRECT READ TESTS")

    for name, args in [
        (
            "get_running_applications",
            {},
        ),
        (
            "get_frontmost_window",
            {},
        ),
        (
            "get_application_windows",
            {
                "application": "Code",
            },
        ),
    ]:

        result = execute_capability(
            name,
            args,
        )

        print(
            f"\n{name}"
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

    divider("3. FULL PIPELINE")

    queries = [
        # "What apps are running?",
        "What's the current window?",
        "List Code windows",
        # "Switch to Code",
        # "Focus Code window",
        "Resize Code window to 900x600",
        "Resize TextEdit window to 900x600",
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