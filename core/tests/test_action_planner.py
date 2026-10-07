from core.actions.planner import plan_action


def show(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"INPUT: {text}")

    result = plan_action(text)

    print(f"UNDERSTOOD: {result.understood}")

    if result.request:
        request = result.request

        print(f"ACTION:      {request.action}")
        print(f"TARGET TYPE: {request.target_type}")
        print(f"TARGET:      {request.target}")
        print(f"ARGUMENTS:   {request.arguments}")
    else:
        print(f"REASON:      {result.reason}")


def main() -> None:
    tests = [
        "Open TextEdit",
        "Launch Safari",
        "Please open Notes",
        "Stella, open Calculator",

        "Open https://apple.com",
        "Open github.com",
        "Go to openai.com",

        "Open ~/Desktop",
        "Open /Applications",
        'Open "/Users/max/Desktop/test file.txt"',

        "Reveal ~/Desktop/test.txt in Finder",
        "Show /Applications in Finder",

        # Should NOT be understood yet.
        "Delete everything",
        "Click the save button",
        "Move all screenshots",
        "Make my Mac levitate",
    ]

    for text in tests:
        show(text)


if __name__ == "__main__":
    main()