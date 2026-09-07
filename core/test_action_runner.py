from core.actions.runner import run_action
import tempfile
from pathlib import Path


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
    show("Open https://apple.com")
    show("Open ~/Desktop")
    show("Show /Applications in Finder")
    with tempfile.TemporaryDirectory() as temp_dir:
        show(
            f"Create a folder named Stella Test in {temp_dir}"
        )

        created = (
            Path(temp_dir)
            / "Stella Test"
        )

        print(
            f"FOLDER EXISTS: {created.exists()}"
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        source_dir = root / "source"
        destination_dir = root / "destination"

        source_dir.mkdir()
        destination_dir.mkdir()

        source_file = (
            source_dir / "report.txt"
        )

        source_file.write_text(
            "Stella move test"
        )

        show(
            f"Move {source_file} to {destination_dir}"
        )

        moved_file = (
            destination_dir / "report.txt"
        )

        print(
            f"SOURCE EXISTS: {source_file.exists()}"
        )

        print(
            f"MOVED FILE EXISTS: {moved_file.exists()}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        source_dir = root / "source"
        destination_dir = root / "destination"

        source_dir.mkdir()
        destination_dir.mkdir()

        first_file = (
            source_dir / "report.pdf"
        )

        second_file = (
            source_dir / "notes.txt"
        )

        first_file.write_text(
            "Report test"
        )

        second_file.write_text(
            "Notes test"
        )

        show(
            "Move report.pdf and notes.txt "
            f"from {source_dir} to {destination_dir}"
        )

        moved_first = (
            destination_dir
            / "report.pdf"
        )

        moved_second = (
            destination_dir
            / "notes.txt"
        )

        print(
            f"FIRST MOVED: {moved_first.exists()}"
        )

        print(
            f"SECOND MOVED: {moved_second.exists()}"
        )

        print(
            f"FIRST SOURCE EXISTS: {first_file.exists()}"
        )

        print(
            f"SECOND SOURCE EXISTS: {second_file.exists()}"
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        source_dir = root / "source"
        destination_dir = root / "destination"

        source_dir.mkdir()
        destination_dir.mkdir()

        first_file = (
            source_dir / "alpha.txt"
        )

        second_file = (
            source_dir / "beta.txt"
        )

        first_file.write_text(
            "Alpha"
        )

        second_file.write_text(
            "Beta"
        )

        show(
            "Copy alpha.txt and beta.txt "
            f"from {source_dir} to {destination_dir}"
        )

        copied_first = (
            destination_dir / "alpha.txt"
        )

        copied_second = (
            destination_dir / "beta.txt"
        )

        print(
            f"FIRST COPY EXISTS: {copied_first.exists()}"
        )

        print(
            f"SECOND COPY EXISTS: {copied_second.exists()}"
        )

        print(
            f"FIRST SOURCE STILL EXISTS: {first_file.exists()}"
        )

        print(
            f"SECOND SOURCE STILL EXISTS: {second_file.exists()}"
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        source_file = (
            root / "old-name.txt"
        )

        source_file.write_text(
            "Stella rename test"
        )

        show(
            f"Rename {source_file} to new-name.txt"
        )

        renamed_file = (
            root / "new-name.txt"
        )

        print(
            f"OLD FILE EXISTS: {source_file.exists()}"
        )

        print(
            f"NEW FILE EXISTS: {renamed_file.exists()}"
        )

if __name__ == "__main__":
    main()