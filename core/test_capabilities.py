from core.capabilities import (
    initialize_capabilities,
)

from core.capabilities.registry import (
    registry,
)

from core.capabilities.resolver import (
    resolve_capabilities,
    resolve_best_capability,
)

from core.capabilities.executor import (
    execute_capability,
)

from core.capabilities.decision import (
    decide_capability,
)
import tempfile
from pathlib import Path

initialize_capabilities()


print()
print("REGISTERED CAPABILITIES")
print("-----------------------")

for capability in registry.all():

    print(
        f"{capability.name:<20} "
        f"family={capability.family:<12} "
        f"risk={capability.risk.value}"
    )


print()
print()
print("RESOLVER TESTS")
print("--------------")


tests = [
    "open an app",
    "launch application",
    "open website",
    "show this file in finder",
    "open folder",
    "delete everything",
]


for query in tests:

    print()
    print(f"Query: {query}")

    matches = resolve_capabilities(
        query
    )

    if not matches:
        print("  No capability found.")
        continue

    for match in matches:

        print(
            f"  {match.capability.name:<20}"
            f" score={match.score:.3f}"
            f" matched='{match.matched_text}'"
        )


print()
print()
print("EXECUTION TEST")
print("--------------")


match = resolve_best_capability(
    "launch app"
)


if match is None:

    print(
        "No matching capability."
    )

else:

    print(
        f"Resolved → "
        f"{match.capability.name}"
    )

    result = execute_capability(
        match.capability.name,
        {
            "application": "TextEdit"
        },
    )

    print(result)


print()
print()
print("FILESYSTEM EXECUTION TEST")
print("-------------------------")

with tempfile.TemporaryDirectory() as temp_dir:
    result = execute_capability(
        "create_folder",
        {
            "parent_path": temp_dir,
            "folder_name": "Stella Test",
        },
    )

    print(result)


print()
print()
print("CAPABILITY DECISION TESTS")
print("-------------------------")

decision_tests = [
    # Existing capabilities
    "open an application",
    "open a website",

    # Should be implementable
    "delete old screenshots",
    "move files into another folder",
    "capture my screen",
    "run a Shortcut",
    "click a button in another application",
    "watch a folder for new files",

    # Deliberately weird / unsupported
    "physically rotate my MacBook ninety degrees",
    "make my Mac levitate",
    "change the color of the physical keyboard keys",
]

print()
print()
print("MOVE FILE EXECUTION TEST")
print("------------------------")

with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)

    source_dir = root / "source"
    destination_dir = root / "destination"

    source_dir.mkdir()
    destination_dir.mkdir()

    source_file = (
        source_dir / "stella-test.txt"
    )

    source_file.write_text(
        "Hello from Stella."
    )

    result = execute_capability(
        "move_file",
        {
            "source_path": str(source_file),
            "destination_path": str(destination_dir),
        },
    )

    moved_file = (
        destination_dir
        / "stella-test.txt"
    )

    print(result)
    print(
        f"SOURCE EXISTS: {source_file.exists()}"
    )
    print(
        f"MOVED FILE EXISTS: {moved_file.exists()}"
    )


for query in decision_tests:

    print()
    print(f"Request: {query}")

    decision = decide_capability(
        query
    )

    print(
        f"  Status: "
        f"{decision.status}"
    )

    if decision.status == "available":

        print(
            f"  Capability: "
            f"{decision.capability.name}"
        )

        print(
            f"  Score: "
            f"{decision.score:.3f}"
        )

    elif decision.gap:

        print(
            f"  Suggested capability: "
            f"{decision.gap.suggested_name}"
        )

        print(
            f"  Family: "
            f"{decision.gap.suggested_family}"
        )

        print(
            f"  Implementation: "
            f"{decision.gap.likely_implementation}"
        )

        print(
            f"  Permissions: "
            f"{decision.gap.required_permissions}"
        )

        print(
            f"  Risk: "
            f"{decision.gap.risk}"
        )

        print(
            f"  Evidence sources:"
        )

        print(
            f"  Confidence: "
            f"{decision.gap.confidence:.3f}"
        )

        for item in decision.gap.evidence[:3]:

            print(
                f"    - "
                f"{item['source']} "
                f"[{item['topic']}]"
                f"distance="
                f"{item['distance']:.2f} "
                f"evidence="
                f"{item['evidence_score']:.3f}"
            )