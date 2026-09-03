"""
workspace.py

Initial low-risk macOS workspace capabilities.

These are Stella's first executable Mac actions.
"""

import subprocess
from pathlib import Path

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)

from core.capabilities.registry import (
    registry,
)


def open_application(
    application: str,
) -> dict:

    subprocess.run(
        [
            "open",
            "-a",
            application,
        ],
        check=True,
    )

    return {
        "application": application,
        "action": "open",
    }


def open_url(
    url: str,
) -> dict:

    subprocess.run(
        [
            "open",
            url,
        ],
        check=True,
    )

    return {
        "url": url,
        "action": "open",
    }


def open_path(
    path: str,
) -> dict:

    target = Path(path).expanduser()

    if not target.exists():
        raise FileNotFoundError(
            f"Path does not exist: "
            f"{target}"
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
    }


def reveal_in_finder(
    path: str,
) -> dict:

    target = Path(path).expanduser()

    if not target.exists():
        raise FileNotFoundError(
            f"Path does not exist: "
            f"{target}"
        )

    subprocess.run(
        [
            "open",
            "-R",
            str(target),
        ],
        check=True,
    )

    return {
        "path": str(target),
        "action": "revealed_in_finder",
    }


def register_workspace_capabilities():

    registry.register(
        Capability(
            name="open_application",

            family="workspace",

            description=(
                "Launch an installed "
                "macOS application."
            ),

            handler=open_application,

            risk=CapabilityRisk.LOW,

            permissions=[],

            requires_confirmation=False,

            supports_undo=False,

            implementation=(
                "macOS LaunchServices / open"
            ),

            intents=[
                "open app",
                "launch app",
                "start application",
            ],
        )
    )

    registry.register(
        Capability(
            name="open_url",

            family="workspace",

            description=(
                "Open a URL using the "
                "default macOS application."
            ),

            handler=open_url,

            risk=CapabilityRisk.LOW,

            permissions=[],

            requires_confirmation=False,

            implementation="LaunchServices",

            intents=[
                "open url",
                "open website",
                "launch link",
            ],
        )
    )

    registry.register(
        Capability(
            name="open_path",

            family="workspace",

            description=(
                "Open a file or directory "
                "using its default macOS app."
            ),

            handler=open_path,

            risk=CapabilityRisk.LOW,

            permissions=[
                "filesystem"
            ],

            requires_confirmation=False,

            implementation="LaunchServices",

            intents=[
                "open file",
                "open folder",
            ],
        )
    )

    registry.register(
        Capability(
            name="reveal_in_finder",

            family="workspace",

            description=(
                "Reveal a file or directory "
                "inside Finder."
            ),

            handler=reveal_in_finder,

            risk=CapabilityRisk.LOW,

            permissions=[
                "filesystem"
            ],

            requires_confirmation=False,

            implementation="Finder / LaunchServices",

            intents=[
                "show in finder",
                "reveal file",
                "reveal folder",
            ],
        )
    )