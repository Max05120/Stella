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
    divider("1. CAPABILITY REGISTRATION")

    initialize_capabilities()

    names = registry.names()

    for name in names:
        print(f"  - {name}")

    required = [
        "get_finder_selection",
        "get_finder_window_path",
        "get_finder_windows",
        "open_finder_location",
    ]

    print("\nFinder capabilities:")

    for name in required:
        if name in names:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} MISSING")


def test_parser():
    divider("2. FINDER PARSER")

    queries = [
        "What do I have selected in Finder?",
        "Where am I in Finder?",
        "What's the current Finder location?",
        "What Finder windows are open?",
        "List the open Finder windows",
        "List Finder windows",
        "Show me the Finder windows",
    ]

    for query in queries:
        print(f"\nQUERY: {query}")

        result = plan_action(query)

        print(f"  understood: {result.understood}")

        if result.request:
            print(
                f"  action:      "
                f"{result.request.action}"
            )
            print(
                f"  target_type: "
                f"{result.request.target_type}"
            )
            print(
                f"  target:      "
                f"{result.request.target}"
            )
            print(
                f"  arguments:   "
                f"{result.request.arguments}"
            )
        else:
            print(
                f"  reason:      {result.reason}"
            )


def test_capabilities_directly():
    divider("3. DIRECT CAPABILITY EXECUTION")

    capabilities = [
        "get_finder_selection",
        "get_finder_window_path",
        "get_finder_windows",
    ]

    for capability in capabilities:

        print(f"\nCAPABILITY: {capability}")

        result = execute_capability(
            capability,
            {},
        )

        print(f"  status:  {result.status}")
        print(f"  message: {result.message}")
        print(f"  data:    {result.data}")
        print(f"  error:   {result.error}")


def test_full_action_pipeline():
    divider("4. FULL ACTION PIPELINE")

    queries = [
        "What do I have selected in Finder?",
        "Where am I in Finder?",
        "What Finder windows are open?",
        "List the open Finder windows",
    ]

    for query in queries:

        print(f"\nQUERY: {query}")

        result = run_action(query)

        print(f"  status:     {result.status}")
        print(f"  capability: {result.capability}")
        print(f"  message:    {result.message}")

        if result.request:
            print(
                f"  request:    "
                f"{result.request.action} / "
                f"{result.request.target_type}"
            )

        if result.execution:
            print(
                f"  execution:  "
                f"{result.execution.status}"
            )
            print(
                f"  data:       "
                f"{result.execution.data}"
            )
            print(
                f"  error:      "
                f"{result.execution.error}"
            )


if __name__ == "__main__":

    print("\nSTELLA FINDER DIAGNOSTIC")

    test_registration()
    test_parser()
    test_capabilities_directly()
    test_full_action_pipeline()

    print("\n")
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)