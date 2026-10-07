from core.actions.runner import (
    run_action,
)

from core.capabilities import (
    initialize_capabilities,
)

from core.capabilities.registry import (
    registry,
)

from core.capabilities.executor import (
    execute_capability,
)


def divider(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def test_registration():

    divider("1. REGISTRATION")

    initialize_capabilities()

    required = [
        "get_focused_ui_element",
        "list_ui_elements",
        "click_ui_element",
        "press_menu_item",
    ]

    names = registry.names()

    for name in required:

        print(
            "✓" if name in names else "✗",
            name,
        )


def test_direct():

    divider("2. DIRECT READ TESTS")

    result = execute_capability(
        "get_focused_ui_element",
        {},
    )

    print(
        "\nget_focused_ui_element"
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

    result = execute_capability(
        "list_ui_elements",
        {
            "application": "TextEdit",
            "max_depth": 6,
            "max_results": 40,
        },
    )

    print(
        "\nlist_ui_elements(TextEdit)"
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

    divider("3. PIPELINE")

    queries = [
        # "What's focused?",
        "List the UI in TextEdit",
        "Press New from the File menu in TextEdit",
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