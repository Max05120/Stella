"""
Low-risk macOS system context and control capabilities.
"""

from __future__ import annotations

import subprocess

from AppKit import NSWorkspace
from AppKit import NSPasteboard, NSPasteboardTypeString

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)

from core.capabilities.registry import registry


# ---------------------------------------------------------
# Frontmost application
# ---------------------------------------------------------

def get_frontmost_application() -> dict:
    workspace = NSWorkspace.sharedWorkspace()

    app = workspace.frontmostApplication()

    if app is None:
        return {
            "name": None,
            "bundle_identifier": None,
            "process_identifier": None,
            "action": "inspect",
            "context": "frontmost_application",
        }

    return {
        "name": app.localizedName(),
        "bundle_identifier": app.bundleIdentifier(),
        "process_identifier": app.processIdentifier(),
        "action": "inspect",
        "context": "frontmost_application",
    }


# ---------------------------------------------------------
# Clipboard
# ---------------------------------------------------------

def get_clipboard() -> dict:
    pasteboard = NSPasteboard.generalPasteboard()

    value = pasteboard.stringForType_(
        NSPasteboardTypeString
    )

    return {
        "text": value,
        "action": "inspect",
        "context": "clipboard",
    }


def set_clipboard(
    text: str,
) -> dict:
    pasteboard = NSPasteboard.generalPasteboard()

    pasteboard.clearContents()

    success = pasteboard.setString_forType_(
        text,
        NSPasteboardTypeString,
    )

    if not success:
        raise RuntimeError(
            "macOS pasteboard rejected the clipboard write."
        )

    return {
        "text": text,
        "action": "write",
        "context": "clipboard",
    }


# ---------------------------------------------------------
# System volume
# ---------------------------------------------------------

def _run_osascript(
    script: str,
) -> str:
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                script,
            ],
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
            f"AppleScript failed: {message}"
        ) from exc

    return result.stdout.strip()


def get_system_volume() -> dict:
    script = """
    set currentVolume to output volume of (get volume settings)
    set mutedState to output muted of (get volume settings)

    return (currentVolume as text) & "|" & (mutedState as text)
    """

    output = _run_osascript(
        script
    )

    parts = output.split("|", 1)

    if len(parts) != 2:
        raise RuntimeError(
            f"Unexpected volume response: {output}"
        )

    volume = int(
        parts[0].strip()
    )

    muted = (
        parts[1].strip().lower()
        == "true"
    )

    return {
        "volume": volume,
        "muted": muted,
        "action": "inspect",
        "context": "system_volume",
    }


def set_system_volume(
    volume: int,
) -> dict:
    if not isinstance(
        volume,
        int,
    ):
        raise TypeError(
            "Volume must be an integer."
        )

    if volume < 0 or volume > 100:
        raise ValueError(
            "Volume must be between 0 and 100."
        )

    _run_osascript(
        f"set volume output volume {volume}"
    )

    return {
        "volume": volume,
        "action": "set",
        "context": "system_volume",
    }


def mute_system_volume() -> dict:
    _run_osascript(
        "set volume with output muted"
    )

    return {
        "muted": True,
        "action": "set",
        "context": "system_volume",
    }


def unmute_system_volume() -> dict:
    _run_osascript(
        "set volume without output muted"
    )

    return {
        "muted": False,
        "action": "set",
        "context": "system_volume",
    }


# ---------------------------------------------------------
# Registration
# ---------------------------------------------------------

def register_system_capabilities():

    registry.register(
        Capability(
            name="get_frontmost_application",
            family="system",
            action="inspect",
            description=(
                "Read the application currently "
                "frontmost on macOS."
            ),
            handler=get_frontmost_application,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "NSWorkspace"
            ),
            intents=[
                "get frontmost application",
                "current app",
                "active application",
                "front app",
            ],
        )
    )

    registry.register(
        Capability(
            name="get_clipboard",
            family="clipboard",
            action="inspect",
            description=(
                "Read text currently stored "
                "in the macOS clipboard."
            ),
            handler=get_clipboard,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "NSPasteboard"
            ),
            intents=[
                "get clipboard",
                "read clipboard",
                "clipboard contents",
                "what is copied",
            ],
        )
    )

    registry.register(
        Capability(
            name="set_clipboard",
            family="clipboard",
            action="write",
            description=(
                "Replace the macOS clipboard "
                "with text."
            ),
            handler=set_clipboard,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "NSPasteboard"
            ),
            intents=[
                "set clipboard",
                "copy text to clipboard",
                "put text on clipboard",
            ],
        )
    )

    registry.register(
        Capability(
            name="get_system_volume",
            family="system",
            action="inspect",
            description=(
                "Read the current macOS output "
                "volume and mute state."
            ),
            handler=get_system_volume,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "AppleScript / get volume settings"
            ),
            intents=[
                "get system volume",
                "current volume",
                "volume level",
            ],
        )
    )

    registry.register(
        Capability(
            name="set_system_volume",
            family="system",
            action="set",
            description=(
                "Set the macOS output volume."
            ),
            handler=set_system_volume,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "AppleScript / set volume"
            ),
            intents=[
                "set system volume",
                "change volume",
                "volume to",
            ],
        )
    )

    registry.register(
        Capability(
            name="mute_system_volume",
            family="system",
            action="set",
            description=(
                "Mute macOS system output."
            ),
            handler=mute_system_volume,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "AppleScript / set volume"
            ),
            intents=[
                "mute system volume",
                "mute audio",
                "mute sound",
            ],
        )
    )

    registry.register(
        Capability(
            name="unmute_system_volume",
            family="system",
            action="set",
            description=(
                "Unmute macOS system output."
            ),
            handler=unmute_system_volume,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "AppleScript / set volume"
            ),
            intents=[
                "unmute system volume",
                "unmute audio",
                "turn sound back on",
            ],
        )
    )