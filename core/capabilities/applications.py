"""
Application-level macOS capabilities.
"""

from __future__ import annotations

from AppKit import NSWorkspace

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)
from core.capabilities.registry import registry


def _running_apps():
    workspace = NSWorkspace.sharedWorkspace()

    return list(
        workspace.runningApplications()
    )


def _find_app(
    application: str,
):
    target = application.strip().lower()

    for app in _running_apps():

        name = (
            app.localizedName()
            or ""
        )

        bundle_id = (
            app.bundleIdentifier()
            or ""
        )

        if (
            name.lower() == target
            or bundle_id.lower() == target
        ):
            return app

    return None


def get_running_applications() -> dict:

    applications = []

    for app in _running_apps():

        name = app.localizedName()

        if not name:
            continue

        applications.append({
            "name": name,
            "bundle_identifier": app.bundleIdentifier(),
            "process_identifier": app.processIdentifier(),
            "active": bool(app.isActive()),
            "hidden": bool(app.isHidden()),
        })

    applications.sort(
        key=lambda item:
            item["name"].lower()
    )

    return {
        "applications": applications,
        "count": len(applications),
        "action": "inspect",
        "context": "running_applications",
    }


def activate_application(
    application: str,
) -> dict:

    app = _find_app(
        application
    )

    if app is None:
        raise ValueError(
            f"Running application not found: "
            f"{application}"
        )

    success = app.activateWithOptions_(
        1 << 1
    )

    if not success:
        raise RuntimeError(
            f"Could not activate {application}."
        )

    return {
        "application": app.localizedName(),
        "bundle_identifier": app.bundleIdentifier(),
        "action": "activate",
    }


def hide_application(
    application: str,
) -> dict:

    app = _find_app(
        application
    )

    if app is None:
        raise ValueError(
            f"Running application not found: "
            f"{application}"
        )

    success = app.hide()

    if not success:
        raise RuntimeError(
            f"Could not hide {application}."
        )

    return {
        "application": app.localizedName(),
        "action": "hide",
    }


def quit_application(
    application: str,
) -> dict:

    app = _find_app(
        application
    )

    if app is None:
        raise ValueError(
            f"Running application not found: "
            f"{application}"
        )

    success = app.terminate()

    if not success:
        raise RuntimeError(
            f"Could not quit {application}."
        )

    return {
        "application": app.localizedName(),
        "action": "quit",
    }


def register_application_capabilities():

    registry.register(
        Capability(
            name="get_running_applications",
            family="applications",
            action="inspect",
            description=(
                "List applications currently "
                "running on macOS."
            ),
            handler=get_running_applications,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation="NSWorkspace",
            intents=[
                "get running applications",
                "list running apps",
                "open applications",
            ],
        )
    )

    registry.register(
        Capability(
            name="activate_application",
            family="applications",
            action="activate",
            description=(
                "Bring a running application "
                "to the foreground."
            ),
            handler=activate_application,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation="NSRunningApplication",
            intents=[
                "activate application",
                "focus application",
                "switch to application",
            ],
        )
    )

    registry.register(
        Capability(
            name="hide_application",
            family="applications",
            action="hide",
            description=(
                "Hide a running macOS application."
            ),
            handler=hide_application,
            risk=CapabilityRisk.LOW,
            permissions=[],
            requires_confirmation=False,
            supports_undo=True,
            implementation="NSRunningApplication",
            intents=[
                "hide application",
                "hide app",
            ],
        )
    )

    registry.register(
        Capability(
            name="quit_application",
            family="applications",
            action="quit",
            description=(
                "Request a running macOS "
                "application to quit."
            ),
            handler=quit_application,
            risk=CapabilityRisk.MEDIUM,
            permissions=[],
            requires_confirmation=False,
            supports_undo=False,
            implementation="NSRunningApplication",
            intents=[
                "quit application",
                "close application",
                "exit application",
            ],
        )
    )