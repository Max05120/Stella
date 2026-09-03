from core.actions.runner import run_action


def show(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"INPUT: {text}")

    result = run_action(text)

    print(f"STATUS:     {result.status}")
    print(f"MESSAGE:    {result.message}")

    if result.request:
        print(f"ACTION:     {result.request.action}")
        print(f"TARGET:     {result.request.target}")
        print(f"TYPE:       {result.request.target_type}")

    if result.capability:
        print(f"CAPABILITY: {result.capability}")

    if result.decision:
        print(f"DECISION:   {result.decision}")

    if result.execution:
        print(f"EXECUTION:  {result.execution}")


def main() -> None:

    # Start with ONE real execution test.
    show("Open TextEdit")


if __name__ == "__main__":
    main()