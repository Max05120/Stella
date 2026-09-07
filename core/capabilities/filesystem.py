"""
filesystem.py

Low-risk filesystem capabilities for Stella.

This module contains executable filesystem actions only.
Natural-language interpretation belongs in core/actions/planner.py.
"""

from pathlib import Path
import shutil
import subprocess

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)
from core.capabilities.registry import registry

def trash_files(
    source_paths: list[str],
) -> dict:
    """
    Move multiple existing files to the macOS Trash.

    All sources are validated before Finder is asked
    to trash anything.
    """

    if not source_paths:
        raise ValueError(
            "At least one source file is required."
        )

    sources = [
        Path(path).expanduser()
        for path in source_paths
    ]

    # Validate everything before modifying anything.
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {source}"
            )

        if not source.is_file():
            raise ValueError(
                f"Source is not a file: {source}"
            )

    script = """
    on run argv
        set targetItems to {}

        repeat with itemPath in argv
            set pathText to contents of itemPath
            set targetItem to POSIX file pathText as alias
            set end of targetItems to targetItem
        end repeat

        tell application "Finder"
            delete targetItems
        end tell
    end run
    """

    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                script,
                *[
                    str(source)
                    for source in sources
                ],
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as exc:
        error_message = (
            exc.stderr.strip()
            or exc.stdout.strip()
            or str(exc)
        )

        raise RuntimeError(
            f"Finder Trash operation failed: "
            f"{error_message}"
        ) from exc

    return {
        "source_paths": [
            str(source)
            for source in sources
        ],
        "file_names": [
            source.name
            for source in sources
        ],
        "count": len(sources),
        "action": "trash",
        "stdout": result.stdout.strip(),
    }

def trash_file(
    source_path: str,
) -> dict:
    """
    Move one existing file to the macOS Trash.
    """

    source = Path(
        source_path
    ).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Source is not a file: {source}"
        )

    script = """
    on run argv
        set itemPath to item 1 of argv
        set targetItem to POSIX file itemPath as alias

        tell application "Finder"
            delete targetItem
        end tell
    end run
    """

    result = subprocess.run(
        [
            "osascript",
            "-e",
            script,
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return {
        "source_path": str(source),
        "file_name": source.name,
        "action": "trash",
        "stdout": result.stdout.strip(),
    }

def create_folder(
    parent_path: str,
    folder_name: str,
) -> dict:
    """
    Create one folder inside an existing parent directory.

    This intentionally does not create missing parent directories.
    """

    parent = Path(parent_path).expanduser()

    if not parent.exists():
        raise FileNotFoundError(
            f"Parent path does not exist: {parent}"
        )

    if not parent.is_dir():
        raise NotADirectoryError(
            f"Parent path is not a directory: {parent}"
        )

    folder_name = folder_name.strip()

    if not folder_name:
        raise ValueError(
            "Folder name cannot be empty."
        )

    if folder_name in {".", ".."}:
        raise ValueError(
            "Invalid folder name."
        )

    if "/" in folder_name:
        raise ValueError(
            "Folder name must be a single directory name."
        )

    target = parent / folder_name

    if target.exists():
        raise FileExistsError(
            f"A file or folder already exists at: {target}"
        )

    target.mkdir()

    return {
        "path": str(target),
        "parent_path": str(parent),
        "folder_name": folder_name,
        "action": "create",
    }

def move_file(
    source_path: str,
    destination_path: str,
) -> dict:
    """
    Move one existing file into an existing destination directory.

    v1 intentionally:
    - handles one file at a time
    - does not overwrite existing items
    - does not create missing destination directories
    """

    source = Path(source_path).expanduser()
    destination = Path(destination_path).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Source is not a file: {source}"
        )

    if not destination.exists():
        raise FileNotFoundError(
            f"Destination path does not exist: {destination}"
        )

    if not destination.is_dir():
        raise NotADirectoryError(
            f"Destination is not a directory: {destination}"
        )

    target = destination / source.name

    if target.exists():
        raise FileExistsError(
            f"An item already exists at: {target}"
        )

    source.rename(target)

    return {
        "source_path": str(source),
        "destination_path": str(destination),
        "target_path": str(target),
        "action": "move",
    }

def move_files(
    source_paths: list[str],
    destination_path: str,
) -> dict:
    """
    Move multiple existing files into one existing destination directory.

    v1 intentionally:
    - handles files only
    - uses one destination directory
    - does not overwrite existing items
    - validates the whole request before moving anything
    """

    destination = Path(
        destination_path
    ).expanduser()

    if not destination.exists():
        raise FileNotFoundError(
            f"Destination path does not exist: {destination}"
        )

    if not destination.is_dir():
        raise NotADirectoryError(
            f"Destination is not a directory: {destination}"
        )

    if not source_paths:
        raise ValueError(
            "At least one source file is required."
        )

    sources = [
        Path(path).expanduser()
        for path in source_paths
    ]

    # ---------------------------------------------------------
    # Validate everything before moving anything.
    # ---------------------------------------------------------

    targets = []

    for source in sources:
        if not source.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {source}"
            )

        if not source.is_file():
            raise ValueError(
                f"Source is not a file: {source}"
            )

        target = (
            destination
            / source.name
        )

        if target.exists():
            raise FileExistsError(
                f"An item already exists at: {target}"
            )

        targets.append(target)

    # Prevent ambiguous requests containing duplicate names.
    target_names = [
        target.name
        for target in targets
    ]

    if len(target_names) != len(set(target_names)):
        raise ValueError(
            "Multiple source files would have the same "
            "name in the destination."
        )

    # ---------------------------------------------------------
    # Execute only after the full request is validated.
    # ---------------------------------------------------------

    moved_files = []

    for source, target in zip(
        sources,
        targets,
    ):
        source.rename(target)

        moved_files.append({
            "source_path": str(source),
            "target_path": str(target),
        })

    return {
        "destination_path": str(destination),
        "moved_files": moved_files,
        "count": len(moved_files),
        "action": "move",
    }

def copy_file(
    source_path: str,
    destination_path: str,
) -> dict:
    """
    Copy one existing file into an existing destination directory.

    v1 intentionally:
    - handles one file at a time
    - does not overwrite existing items
    - does not create missing destination directories
    """

    source = Path(source_path).expanduser()
    destination = Path(destination_path).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Source is not a file: {source}"
        )

    if not destination.exists():
        raise FileNotFoundError(
            f"Destination path does not exist: {destination}"
        )

    if not destination.is_dir():
        raise NotADirectoryError(
            f"Destination is not a directory: {destination}"
        )

    target = destination / source.name

    if target.exists():
        raise FileExistsError(
            f"An item already exists at: {target}"
        )

    shutil.copy2(
        source,
        target,
    )

    return {
        "source_path": str(source),
        "destination_path": str(destination),
        "target_path": str(target),
        "action": "copy",
    }


def copy_files(
    source_paths: list[str],
    destination_path: str,
) -> dict:
    """
    Copy multiple existing files into one destination directory.

    Validates everything before copying anything.
    """

    destination = Path(
        destination_path
    ).expanduser()

    if not destination.exists():
        raise FileNotFoundError(
            f"Destination path does not exist: {destination}"
        )

    if not destination.is_dir():
        raise NotADirectoryError(
            f"Destination is not a directory: {destination}"
        )

    if not source_paths:
        raise ValueError(
            "At least one source file is required."
        )

    sources = [
        Path(path).expanduser()
        for path in source_paths
    ]

    targets = []

    for source in sources:
        if not source.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {source}"
            )

        if not source.is_file():
            raise ValueError(
                f"Source is not a file: {source}"
            )

        target = destination / source.name

        if target.exists():
            raise FileExistsError(
                f"An item already exists at: {target}"
            )

        targets.append(target)

    target_names = [
        target.name
        for target in targets
    ]

    if len(target_names) != len(set(target_names)):
        raise ValueError(
            "Multiple source files would have the same "
            "name in the destination."
        )

    copied_files = []

    for source, target in zip(
        sources,
        targets,
    ):
        shutil.copy2(
            source,
            target,
        )

        copied_files.append({
            "source_path": str(source),
            "target_path": str(target),
        })

    return {
        "destination_path": str(destination),
        "copied_files": copied_files,
        "count": len(copied_files),
        "action": "copy",
    }

def rename_file(
    source_path: str,
    new_name: str,
) -> dict:
    """
    Rename one existing file within its current directory.

    v1 intentionally:
    - handles files only
    - keeps the file in the same directory
    - does not overwrite an existing item
    """

    source = Path(source_path).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"Source path does not exist: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Source is not a file: {source}"
        )

    new_name = new_name.strip()

    if not new_name:
        raise ValueError(
            "New file name cannot be empty."
        )

    if new_name in {".", ".."}:
        raise ValueError(
            "Invalid file name."
        )

    if "/" in new_name:
        raise ValueError(
            "New name must be a single file name."
        )

    target = source.parent / new_name

    if target == source:
        raise ValueError(
            "The new file name is the same as the current name."
        )

    if target.exists():
        raise FileExistsError(
            f"An item already exists at: {target}"
        )

    source.rename(target)

    return {
        "source_path": str(source),
        "target_path": str(target),
        "old_name": source.name,
        "new_name": new_name,
        "action": "rename",
    }


def register_filesystem_capabilities():
    registry.register(
        Capability(
            name="trash_file",
            family="filesystem",
            action="delete",
            description=(
                "Move one existing file to "
                "the macOS Trash."
            ),
            handler=trash_file,
            risk=CapabilityRisk.DESTRUCTIVE,
            permissions=[
                "filesystem",
                "automation",
            ],
            requires_confirmation=True,
            supports_undo=False,
            implementation=(
                "Finder AppleScript / macOS Trash"
            ),
            intents=[
                "trash file",
                "move file to trash",
                "put file in trash",
            ],
        )
    )

    registry.register(
        Capability(
            name="trash_files",
            family="filesystem",
            action="delete",
            description=(
                "Move multiple existing files "
                "to the macOS Trash."
            ),
            handler=trash_files,
            risk=CapabilityRisk.DESTRUCTIVE,
            permissions=[
                "filesystem",
                "automation",
            ],
            requires_confirmation=True,
            supports_undo=False,
            implementation=(
                "Finder AppleScript / macOS Trash"
            ),
            intents=[
                "trash files",
                "move files to trash",
                "put files in trash",
            ],
        )
    )

    registry.register(
        Capability(
            name="create_folder",
            family="filesystem",
            action="create",
            description=(
                "Create a new folder inside an "
                "existing filesystem directory."
            ),
            handler=create_folder,
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
                "create folder",
                "make folder",
                "new folder",
                "create directory",
            ],
        )
    )
    
    registry.register(
        Capability(
            name="move_file",
            family="filesystem",
            action="move",
            description=(
                "Move one existing file into an "
                "existing filesystem directory."
            ),
            handler=move_file,
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
                "move file",
                "move a file",
                "relocate file",
            ],
        )
    )

    registry.register(
        Capability(
            name="move_files",
            family="filesystem",
            action="move",
            description=(
                "Move multiple existing files into "
                "one existing filesystem directory."
            ),
               
            handler=move_files,
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
                "move files",
                "move multiple files",
                "relocate files",
            ],
        )
    )

    registry.register(
        Capability(
            name="copy_file",
            family="filesystem",
            action="copy",
            description=(
                "Copy one existing file into an "
                "existing filesystem directory."
            ),
            handler=copy_file,
            risk=CapabilityRisk.LOW,
            permissions=[
                "filesystem",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "Python shutil / macOS filesystem"
            ),
            intents=[
                "copy file",
                "copy a file",
                "duplicate file",
            ],
        )
    )

    registry.register(
        Capability(
            name="copy_files",
            family="filesystem",
            action="copy",
            description=(
                "Copy multiple existing files into "
                "one existing filesystem directory."
            ),
            handler=copy_files,
            risk=CapabilityRisk.LOW,
            permissions=[
                "filesystem",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "Python shutil / macOS filesystem"
            ),
            intents=[
                "copy files",
                "copy multiple files",
                "duplicate files",
            ],
        )
    )
    registry.register(
        Capability(
            name="rename_file",
            family="filesystem",
            action="rename",
            description=(
                "Rename one existing file within "
                "its current filesystem directory."
            ),
            handler=rename_file,
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
                "rename file",
                "rename a file",
                "change file name",
            ],
        )
    )

    