"""
macOS window inspection and control capabilities.

Uses the Accessibility API.
"""

from __future__ import annotations

from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    AXUIElementSetAttributeValue,
    AXUIElementIsAttributeSettable,
    AXUIElementPerformAction,
    AXValueCreate,
    AXValueGetValue,
    kAXWindowsAttribute,
    kAXFocusedWindowAttribute,
    kAXTitleAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    kAXRaiseAction,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

from AppKit import NSWorkspace
from Quartz import CGPoint, CGSize

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)
from core.capabilities.registry import registry


def _find_running_application(
    application: str,
):
    target = application.strip().lower()

    workspace = (
        NSWorkspace.sharedWorkspace()
    )

    for app in workspace.runningApplications():

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


def _copy_attribute(
    element,
    attribute,
):
    try:
        error, value = (
            AXUIElementCopyAttributeValue(
                element,
                attribute,
                None,
            )
        )

        if error != 0:
            return None

        return value

    except Exception as exc:
        raise RuntimeError(
            f"Could not read AX attribute "
            f"{attribute}: {exc}"
        ) from exc


def _application_ax(
    application: str,
):
    app = _find_running_application(
        application
    )

    if app is None:
        raise ValueError(
            f"Running application not found: "
            f"{application}"
        )

    return (
        app,
        AXUIElementCreateApplication(
            app.processIdentifier()
        ),
    )


def _point_from_ax(
    value,
):
    if value is None:
        return None

    result = CGPoint()

    success = AXValueGetValue(
        value,
        kAXValueCGPointType,
        result,
    )

    if not success:
        return None

    return {
        "x": result.x,
        "y": result.y,
    }


def _size_from_ax(
    value,
):
    if value is None:
        return None

    result = CGSize()

    success = AXValueGetValue(
        value,
        kAXValueCGSizeType,
        result,
    )

    if not success:
        return None

    return {
        "width": result.width,
        "height": result.height,
    }


def _window_info(
    window,
    index: int,
) -> dict:

    title = _copy_attribute(
        window,
        kAXTitleAttribute,
    )

    return {
        "index": index,
        "title": (
            str(title)
            if title is not None
            else ""
        ),
    }


def get_application_windows(
    application: str,
) -> dict:

    app, ax_app = _application_ax(
        application
    )

    windows = (
        _copy_attribute(
            ax_app,
            kAXWindowsAttribute,
        )
        or []
    )

    results = [
        _window_info(
            window,
            index,
        )
        for index, window in enumerate(
            windows
        )
    ]

    return {
        "application": app.localizedName(),
        "windows": results,
        "count": len(results),
        "action": "inspect",
        "context": "application_windows",
    }


def get_frontmost_window() -> dict:

    workspace = (
        NSWorkspace.sharedWorkspace()
    )

    app = (
        workspace.frontmostApplication()
    )

    if app is None:
        return {
            "application": None,
            "window": None,
            "action": "inspect",
            "context": "frontmost_window",
        }

    ax_app = AXUIElementCreateApplication(
        app.processIdentifier()
    )

    window = _copy_attribute(
        ax_app,
        kAXFocusedWindowAttribute,
    )

    if window is None:
        return {
            "application": app.localizedName(),
            "window": None,
            "action": "inspect",
            "context": "frontmost_window",
        }

    return {
        "application": app.localizedName(),
        "window": _window_info(
            window,
            0,
        ),
        "action": "inspect",
        "context": "frontmost_window",
    }


def _window_by_index(
    application: str,
    window_index: int,
):

    app, ax_app = _application_ax(
        application
    )

    windows = (
        _copy_attribute(
            ax_app,
            kAXWindowsAttribute,
        )
        or []
    )

    if (
        window_index < 0
        or window_index >= len(windows)
    ):
        raise IndexError(
            f"Window index {window_index} "
            f"is not available for {application}."
        )

    return app, windows[window_index]


def focus_window(
    application: str,
    window_index: int = 0,
) -> dict:

    app, window = _window_by_index(
        application,
        window_index,
    )

    app.activateWithOptions_(
        1 << 1
    )

    error = AXUIElementPerformAction(
        window,
        kAXRaiseAction,
    )

    if error != 0:
        raise RuntimeError(
            f"Could not focus window "
            f"{window_index}."
        )

    return {
        "application": app.localizedName(),
        "window_index": window_index,
        "action": "focus",
    }


def move_window(
    application: str,
    x: int,
    y: int,
    window_index: int = 0,
) -> dict:

    app, window = _window_by_index(
        application,
        window_index,
    )

    point = CGPoint(
        x,
        y,
    )

    value = AXValueCreate(
        kAXValueCGPointType,
        point,
    )

    error = AXUIElementSetAttributeValue(
        window,
        kAXPositionAttribute,
        value,
    )

    if error != 0:
        raise RuntimeError(
            "Could not move the window."
        )

    return {
        "application": app.localizedName(),
        "window_index": window_index,
        "x": x,
        "y": y,
        "action": "move",
    }


def resize_window(
    application: str,
    width: int,
    height: int,
    window_index: int = 0,
) -> dict:

    if width <= 0 or height <= 0:
        raise ValueError(
            "Window width and height must be positive."
        )

    app, window = _window_by_index(
        application,
        window_index,
    )

    check_error, settable = (
        AXUIElementIsAttributeSettable(
            window,
            kAXSizeAttribute,
            None,
        )
    )

    if check_error != 0:
        raise RuntimeError(
            f"Could not inspect window resize support. "
            f"AX error: {check_error}"
        )

    if not settable:
        return {
            "application": app.localizedName(),
            "window_index": window_index,
            "width": width,
            "height": height,
            "resized": False,
            "reason": "size_not_settable",
            "action": "resize",
        }

    size = CGSize(
        width,
        height,
    )

    value = AXValueCreate(
        kAXValueCGSizeType,
        size,
    )

    if value is None:
        raise RuntimeError(
            "Could not create AX size value."
        )

    error = AXUIElementSetAttributeValue(
        window,
        kAXSizeAttribute,
        value,
    )

    if error != 0:
        raise RuntimeError(
            f"Could not resize the window. "
            f"AX error: {error}"
        )

    return {
        "application": app.localizedName(),
        "window_index": window_index,
        "width": width,
        "height": height,
        "action": "resize",
    }
   


def register_window_capabilities():

    registry.register(
        Capability(
            name="get_application_windows",
            family="windows",
            action="inspect",
            description=(
                "List windows belonging to "
                "a running macOS application."
            ),
            handler=get_application_windows,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation="AXUIElement",
            intents=[
                "get application windows",
                "list app windows",
            ],
        )
    )

    registry.register(
        Capability(
            name="get_frontmost_window",
            family="windows",
            action="inspect",
            description=(
                "Read information about the "
                "current frontmost window."
            ),
            handler=get_frontmost_window,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation="AXUIElement",
            intents=[
                "get frontmost window",
                "current window",
                "active window",
            ],
        )
    )

    registry.register(
        Capability(
            name="focus_window",
            family="windows",
            action="focus",
            description=(
                "Bring one application window "
                "to the foreground."
            ),
            handler=focus_window,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation="AXUIElement",
            intents=[
                "focus window",
                "raise window",
            ],
        )
    )

    registry.register(
        Capability(
            name="move_window",
            family="windows",
            action="move",
            description=(
                "Move an application window "
                "to screen coordinates."
            ),
            handler=move_window,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=True,
            implementation="AXUIElement",
            intents=[
                "move window",
                "reposition window",
            ],
        )
    )

    registry.register(
        Capability(
            name="resize_window",
            family="windows",
            action="resize",
            description=(
                "Resize an application window."
            ),
            handler=resize_window,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=True,
            implementation="AXUIElement",
            intents=[
                "resize window",
                "change window size",
            ],
        )
    )