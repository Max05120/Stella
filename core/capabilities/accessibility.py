"""
macOS Accessibility/UI interaction capabilities.

Initial scope:
- inspect focused UI element
- inspect application controls
- click/press a control by accessible label
- press an application menu item
"""

from __future__ import annotations

import subprocess

from AppKit import NSWorkspace

from ApplicationServices import (
    AXIsProcessTrusted,
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyActionNames,
    AXUIElementPerformAction,

    kAXFocusedUIElementAttribute,
    kAXChildrenAttribute,
    kAXRoleAttribute,
    kAXTitleAttribute,
    kAXDescriptionAttribute,
    kAXValueAttribute,
    kAXEnabledAttribute,
    kAXPressAction,
)

from core.capabilities.models import (
    Capability,
    CapabilityRisk,
)

from core.capabilities.registry import registry


# ---------------------------------------------------------
# Common helpers
# ---------------------------------------------------------

def _require_accessibility():
    if not AXIsProcessTrusted():
        raise PermissionError(
            "Stella does not have macOS Accessibility permission."
        )


def _running_application(
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

    raise ValueError(
        f"Running application not found: {application}"
    )


def _ax_application(
    application: str,
):
    app = _running_application(
        application
    )

    ax_app = (
        AXUIElementCreateApplication(
            app.processIdentifier()
        )
    )

    return app, ax_app


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

    except Exception:
        return None

    if error != 0:
        return None

    return value


def _copy_actions(
    element,
) -> list[str]:
    try:
        error, actions = (
            AXUIElementCopyActionNames(
                element,
                None,
            )
        )

    except Exception:
        return []

    if error != 0 or not actions:
        return []

    return [
        str(action)
        for action in actions
    ]


def _safe_text(
    value,
) -> str | None:

    if value is None:
        return None

    try:
        text = str(value).strip()
    except Exception:
        return None

    return text or None


def _element_info(
    element,
) -> dict:

    role = _safe_text(
        _copy_attribute(
            element,
            kAXRoleAttribute,
        )
    )

    title = _safe_text(
        _copy_attribute(
            element,
            kAXTitleAttribute,
        )
    )

    description = _safe_text(
        _copy_attribute(
            element,
            kAXDescriptionAttribute,
        )
    )

    value = _safe_text(
        _copy_attribute(
            element,
            kAXValueAttribute,
        )
    )

    enabled_value = _copy_attribute(
        element,
        kAXEnabledAttribute,
    )

    enabled = (
        bool(enabled_value)
        if enabled_value is not None
        else None
    )

    return {
        "role": role,
        "title": title,
        "description": description,
        "value": value,
        "enabled": enabled,
        "actions": _copy_actions(
            element
        ),
    }


# ---------------------------------------------------------
# Focused UI element
# ---------------------------------------------------------

def get_focused_ui_element() -> dict:
    """
    Inspect the focused Accessibility element
    of the current frontmost application.
    """

    _require_accessibility()

    workspace = (
        NSWorkspace.sharedWorkspace()
    )

    app = (
        workspace.frontmostApplication()
    )

    if app is None:
        return {
            "application": None,
            "element": None,
            "action": "inspect",
            "context": "focused_ui_element",
        }

    ax_app = (
        AXUIElementCreateApplication(
            app.processIdentifier()
        )
    )

    focused = _copy_attribute(
        ax_app,
        kAXFocusedUIElementAttribute,
    )

    if focused is None:
        return {
            "application":
                app.localizedName(),
            "element": None,
            "action": "inspect",
            "context": "focused_ui_element",
        }

    return {
        "application":
            app.localizedName(),

        "element":
            _element_info(
                focused
            ),

        "action": "inspect",
        "context": "focused_ui_element",
    }

# ---------------------------------------------------------
# UI tree inspection
# ---------------------------------------------------------

def _walk_elements(
    element,
    depth: int,
    max_depth: int,
    results: list,
    max_results: int,
):
    if len(results) >= max_results:
        return

    info = _element_info(
        element
    )

    if (
        info.get("title")
        or info.get("description")
        or info.get("value")
        or info.get("actions")
    ):
        results.append(
            info
        )

    if depth >= max_depth:
        return

    children = (
        _copy_attribute(
            element,
            kAXChildrenAttribute,
        )
        or []
    )

    for child in children:

        if len(results) >= max_results:
            break

        _walk_elements(
            child,
            depth + 1,
            max_depth,
            results,
            max_results,
        )


def list_ui_elements(
    application: str,
    max_depth: int = 6,
    max_results: int = 100,
) -> dict:
    """
    Return a bounded snapshot of accessible controls.
    """

    _require_accessibility()

    app, ax_app = _ax_application(
        application
    )

    results = []

    _walk_elements(
        ax_app,
        depth=0,
        max_depth=max_depth,
        results=results,
        max_results=max_results,
    )

    return {
        "application": app.localizedName(),
        "elements": results,
        "count": len(results),
        "action": "inspect",
        "context": "ui_elements",
    }


# ---------------------------------------------------------
# Find / press UI control
# ---------------------------------------------------------

def _matches_label(
    info: dict,
    label: str,
) -> bool:

    target = label.strip().lower()

    candidates = [
        info.get("title"),
        info.get("description"),
        info.get("value"),
    ]

    for candidate in candidates:

        if not candidate:
            continue

        normalized = (
            str(candidate)
            .strip()
            .lower()
        )

        if normalized == target:
            return True

    return False


def _find_element_by_label(
    root,
    label: str,
    depth: int = 0,
    max_depth: int = 8,
):
    info = _element_info(
        root
    )

    if _matches_label(
        info,
        label,
    ):
        return root, info

    if depth >= max_depth:
        return None

    children = (
        _copy_attribute(
            root,
            kAXChildrenAttribute,
        )
        or []
    )

    for child in children:

        result = (
            _find_element_by_label(
                child,
                label,
                depth + 1,
                max_depth,
            )
        )

        if result is not None:
            return result

    return None


def click_ui_element(
    application: str,
    label: str,
) -> dict:
    """
    Find an accessible UI element by exact label
    and perform AXPress on it.
    """

    _require_accessibility()

    app, ax_app = _ax_application(
        application
    )

    found = _find_element_by_label(
        ax_app,
        label,
    )

    if found is None:
        raise ValueError(
            f"No accessible UI element named "
            f"'{label}' was found in {application}."
        )

    element, info = found

    actions = _copy_actions(
        element
    )

    if str(kAXPressAction) not in actions:
        raise RuntimeError(
            f"The UI element '{label}' does not "
            f"support the press action."
        )

    error = AXUIElementPerformAction(
        element,
        kAXPressAction,
    )

    if error != 0:
        raise RuntimeError(
            f"Could not press '{label}'. "
            f"AX error: {error}"
        )

    return {
        "application": app.localizedName(),
        "label": label,
        "element": info,
        "action": "click",
    }


# ---------------------------------------------------------
# Menu item
# ---------------------------------------------------------

def press_menu_item(
    application: str,
    menu_item: str,
    menu_name: str | None = None,
) -> dict:
    """
    Press an application menu item through System Events.

    Example:
        application = "TextEdit"
        menu_name = "File"
        menu_item = "New"
    """

    _require_accessibility()

    if menu_name:
        script = """
        on run argv
            set appName to item 1 of argv
            set menuName to item 2 of argv
            set itemName to item 3 of argv

            tell application "System Events"
                tell process appName
                    click menu item itemName of menu menuName of menu bar 1
                end tell
            end tell
        end run
        """

        arguments = [
            application,
            menu_name,
            menu_item,
        ]

    else:
        script = """
        on run argv
            set appName to item 1 of argv
            set itemName to item 2 of argv

            tell application "System Events"
                tell process appName
                    repeat with currentMenu in menus of menu bar 1
                        try
                            if exists menu item itemName of currentMenu then
                                click menu item itemName of currentMenu
                                return
                            end if
                        end try
                    end repeat
                end tell
            end tell

            error "Menu item not found"
        end run
        """

        arguments = [
            application,
            menu_item,
        ]

    command = [
        "osascript",
        "-e",
        script,
        *arguments,
    ]

    try:
        subprocess.run(
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
            f"Could not press menu item "
            f"'{menu_item}': {message}"
        ) from exc

    return {
        "application": application,
        "menu": menu_name,
        "menu_item": menu_item,
        "action": "press",
    }


# ---------------------------------------------------------
# Registration
# ---------------------------------------------------------

def register_accessibility_capabilities():

    registry.register(
        Capability(
            name="get_focused_ui_element",
            family="accessibility",
            action="inspect",
            description=(
                "Inspect the currently focused "
                "macOS Accessibility UI element."
            ),
            handler=get_focused_ui_element,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation="AXUIElement",
            intents=[
                "get focused ui element",
                "current focused control",
                "focused field",
            ],
        )
    )

    registry.register(
        Capability(
            name="list_ui_elements",
            family="accessibility",
            action="inspect",
            description=(
                "List accessible UI controls "
                "for a running application."
            ),
            handler=list_ui_elements,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation="AXUIElement",
            intents=[
                "list ui elements",
                "show app controls",
                "inspect app interface",
            ],
        )
    )

    registry.register(
        Capability(
            name="click_ui_element",
            family="accessibility",
            action="click",
            description=(
                "Press an accessible UI control "
                "identified by its label."
            ),
            handler=click_ui_element,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation="AXUIElement / AXPress",
            intents=[
                "click ui element",
                "press button",
                "click button",
            ],
        )
    )

    registry.register(
        Capability(
            name="press_menu_item",
            family="accessibility",
            action="press",
            description=(
                "Press an application menu item."
            ),
            handler=press_menu_item,
            risk=CapabilityRisk.LOW,
            permissions=[
                "accessibility",
                "automation",
            ],
            requires_confirmation=False,
            supports_undo=False,
            implementation=(
                "System Events AppleScript"
            ),
            intents=[
                "press menu item",
                "click menu item",
                "choose menu item",
            ],
        )
    )