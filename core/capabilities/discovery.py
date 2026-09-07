"""
Read-only filesystem discovery capabilities for Stella.

These capabilities inspect the filesystem but do not modify it.

Natural-language interpretation belongs in
core/actions/parsers/discovery.py.
"""

from pathlib import Path

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)

from core.capabilities.registry import registry


def list_files(
    directory_path: str,
    extension: str | None = None,
) -> dict:
    """
    List files directly inside one existing directory.

    v1 intentionally:
    - files only
    - non-recursive
    - optional extension filter
    - does not modify anything
    """

    directory = Path(
        directory_path
    ).expanduser()

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {directory}"
        )

    normalized_extension = None

    if extension:
        normalized_extension = (
            extension.strip().lower()
        )

        if not normalized_extension.startswith("."):
            normalized_extension = (
                f".{normalized_extension}"
            )

    files = []

    for item in directory.iterdir():

        if not item.is_file():
            continue

        if (
            normalized_extension
            and item.suffix.lower()
            != normalized_extension
        ):
            continue

        files.append(item)

    files.sort(
        key=lambda path: path.name.lower()
    )

    return {
        "directory_path": str(directory),
        "extension": normalized_extension,
        "paths": [
            str(path)
            for path in files
        ],
        "file_names": [
            path.name
            for path in files
        ],
        "count": len(files),
        "action": "list",
    }


def register_discovery_capabilities():
    registry.register(
        Capability(
            name="list_files",
            family="discovery",
            action="list",
            description=(
                "List files inside an existing "
                "filesystem directory."
            ),
            handler=list_files,
            risk=CapabilityRisk.LOW,
            permissions=[
                "filesystem",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "Python pathlib / macOS filesystem"
            ),
            intents=[
                "list files",
                "show files",
                "find files in folder",
                "files in folder",
            ],
        )
    )