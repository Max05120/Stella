"""
Finder context and navigation capabilities for Stella.

These capabilities inspect or navigate Finder.
They do not perform destructive filesystem mutations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)

from core.capabilities.registry import registry


def _run_applescript(
    script: str,
    arguments: list[str] | None = None,
) -> str:

    command = [
        "osascript",
        "-e",
        script,
    ]

    if arguments:
        command.extend(arguments)

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as exc:

        message = (
            exc.stderr.strip()
            or exc.stdout.strip()
            or str(exc)
        )

        raise RuntimeError(
            f"Finder AppleScript failed: {message}"
        ) from exc

    return result.stdout.strip()


def get_finder_selection() -> dict:
    """
    Return selected Finder items as POSIX paths.
    """

    script = """
    tell application "Finder"
        set selectedItems to selection

        set outputLines to {}

        repeat with selectedItem in selectedItems
            try
                set itemPath to POSIX path of (selectedItem as alias)
                set end of outputLines to itemPath
            end try
        end repeat

        set AppleScript's text item delimiters to linefeed
        return outputLines as text
    end tell
    """

    output = _run_applescript(
        script
    )

    paths = (
        [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ]
        if output
        else []
    )

    return {
        "paths": paths,
        "count": len(paths),
        "action": "inspect",
        "context": "finder_selection",
    }


def get_finder_window_path() -> dict:
    """
    Return the location of the frontmost Finder window.

    If no Finder window is open, return None.
    """

    script = """
    tell application "Finder"

        if (count of Finder windows) is 0 then
            return ""
        end if

        set windowTarget to target of front Finder window

        return POSIX path of (windowTarget as alias)

    end tell
    """

    output = _run_applescript(
        script
    )

    path = (
        output.strip()
        if output.strip()
        else None
    )

    return {
        "path": path,
        "action": "inspect",
        "context": "finder_window_path",
    }


def get_finder_windows() -> dict:
    """
    Return paths for currently open Finder windows.
    """

    script = """
    tell application "Finder"

        set outputLines to {}

        set windowCount to count of Finder windows

        repeat with windowIndex from 1 to windowCount

            try
                set windowTarget to target of Finder window windowIndex

                set targetAlias to windowTarget as alias

                set windowPath to POSIX path of targetAlias

                set end of outputLines to windowPath
            end try

        end repeat

        set AppleScript's text item delimiters to linefeed

        return outputLines as text

    end tell
    """

    output = _run_applescript(
        script
    )

    paths = (
        [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ]
        if output
        else []
    )

    return {
        "paths": paths,
        "count": len(paths),
        "action": "inspect",
        "context": "finder_windows",
    }


def open_finder_location(
    path: str,
) -> dict:
    """
    Open an existing directory in Finder.
    """

    target = Path(
        path
    ).expanduser()

    if not target.exists():
        raise FileNotFoundError(
            f"Path does not exist: {target}"
        )

    if not target.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {target}"
        )

    subprocess.run(
        [
            "open",
            str(target),
        ],
        check=True,
    )

    return {
        "path": str(target),
        "action": "open",
        "context": "finder_location",
    }


def register_finder_capabilities():

    registry.register(
        Capability(
            name="get_finder_selection",
            family="finder",
            action="inspect",
            description=(
                "Read the files and folders currently "
                "selected in Finder."
            ),
            handler=get_finder_selection,
            risk=CapabilityRisk.LOW,
            permissions=[
                "automation",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "Finder AppleScript / Apple Events"
            ),
            intents=[
                "get finder selection",
                "selected files in finder",
                "selected file in finder",
                "current finder selection",
            ],
        )
    )

    registry.register(
        Capability(
            name="get_finder_window_path",
            family="finder",
            action="inspect",
            description=(
                "Read the current directory of the "
                "frontmost Finder window."
            ),
            handler=get_finder_window_path,
            risk=CapabilityRisk.LOW,
            permissions=[
                "automation",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "Finder AppleScript / Apple Events"
            ),
            intents=[
                "current finder folder",
                "finder window path",
                "where am i in finder",
                "front finder window",
            ],
        )
    )

    registry.register(
        Capability(
            name="get_finder_windows",
            family="finder",
            action="inspect",
            description=(
                "List the directories currently open "
                "in Finder windows."
            ),
            handler=get_finder_windows,
            risk=CapabilityRisk.LOW,
            permissions=[
                "automation",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "Finder AppleScript / Apple Events"
            ),
            intents=[
                "list finder windows",
                "open finder windows",
                "finder windows",
            ],
        )
    )

    registry.register(
        Capability(
            name="open_finder_location",
            family="finder",
            action="open",
            description=(
                "Open an existing filesystem directory "
                "in Finder."
            ),
            handler=open_finder_location,
            risk=CapabilityRisk.LOW,
            permissions=[
                "filesystem",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "macOS LaunchServices / open"
            ),
            intents=[
                "open folder in finder",
                "open finder location",
                "show folder in finder",
            ],
        )
    )