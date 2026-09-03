"""
fetch_docs.py

Downloads curated macOS technical references into data/mac_raw/.

Sources are tagged by:
- topic
- authority
- authority_score
- source_url
- publisher

The resulting text files feed the dedicated Mac knowledge pipeline:

fetch_docs.py
    ↓
ingest_mac.py
    ↓
chunk_mac.py
    ↓
embed_mac.py
    ↓
Chroma collection: mac_knowledge
"""

from pathlib import Path
from urllib.parse import urlparse

import trafilatura
import requests


RAW_DIR = Path("data/mac_raw")


SOURCES = {
    # ---------------------------------------------------------
    # AppleScript / JXA
    # ---------------------------------------------------------

    "mac_automation_scripting_guide.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/"
            "AboutthisGuide.html",

        "topic": "applescript",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },
    "accessibility_programming_guide.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "Accessibility/Conceptual/AccessibilityMacOSX/index.html",

        "topic": "accessibility",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },

    "accessibility_model.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "Accessibility/Conceptual/AccessibilityMacOSX/"
            "OSXAXmodel.html",

        "topic": "accessibility",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },

    "accessibility_custom_controls.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "Accessibility/Conceptual/AccessibilityMacOSX/"
            "ImplementingAccessibilityforCustomControls.html",

        "topic": "accessibility",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },

    "filemanager_reference.txt": {
        "url":
            "https://developer.apple.com/documentation/foundation/filemanager",

        "topic": "filesystem",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "url_resource_values.txt": {
        "url":
            "https://developer.apple.com/documentation/foundation/urlresourcevalues",

        "topic": "filesystem",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
        },

    "how_mac_scripting_works.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/"
            "HowMacScriptingWorks.html",

        "topic": "applescript",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },

    "scripting_bridge_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/scriptingbridge",

        "topic": "scripting_bridge",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "jxa_cookbook_readme.txt": {
        "url":
            "https://github.com/JXA-Cookbook/JXA-Cookbook",

        "topic": "jxa",
        "publisher": "JXA Cookbook",
        "authority": "community",
        "authority_score": 0.65,
    },

    # ---------------------------------------------------------
    # EventKit
    # ---------------------------------------------------------

    "eventkit_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/eventkit",

        "topic": "eventkit",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "eventkit_retrieving_events.txt": {
        "url":
            "https://developer.apple.com/documentation/eventkit/"
            "retrieving-events-and-reminders",

        "topic": "eventkit",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "eventkit_creating_events.txt": {
        "url":
            "https://developer.apple.com/documentation/eventkit/"
            "creating-events-and-reminders",

        "topic": "eventkit",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    # ---------------------------------------------------------
    # Shortcuts
    # ---------------------------------------------------------

    "shortcuts_cli.txt": {
        "url":
            "https://support.apple.com/guide/shortcuts-mac/"
            "run-shortcuts-from-the-command-line-apd455c82f02/mac",

        "topic": "shortcuts",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 0.98,
    },

    "shortcuts_url_schemes_intro.txt": {
        "url":
            "https://support.apple.com/guide/shortcuts-mac/"
            "apd621a1ad7a/mac",

        "topic": "shortcuts",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 0.98,
    },

    "shortcuts_run_via_url.txt": {
        "url":
            "https://support.apple.com/guide/shortcuts-mac/"
            "run-a-shortcut-from-a-url-apd624386f42/mac",

        "topic": "shortcuts",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 0.98,
    },

    # ---------------------------------------------------------
    # launchd
    # ---------------------------------------------------------

    "launchd_tutorial.txt": {
        "url":
            "https://www.launchd.info/",

        "topic": "launchd",
        "publisher": "launchd.info",
        "authority": "community",
        "authority_score": 0.60,
    },

    "launchd_agent_example.txt": {
        "url":
            "https://thoughtbot.com/blog/"
            "example-writing-a-launch-agent-for-apples-launchd",

        "topic": "launchd",
        "publisher": "thoughtbot",
        "authority": "community",
        "authority_score": 0.55,
    },

    # ---------------------------------------------------------
    # Accessibility / UI Automation
    # ---------------------------------------------------------

    "accessibility_api_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/accessibility/"
            "accessibility-api",

        "topic": "accessibility",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "axuielement_reference.txt": {
        "url":
            "https://developer.apple.com/documentation/"
            "applicationservices/axuielement_h",

        "topic": "accessibility",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    # ---------------------------------------------------------
    # NSWorkspace
    # ---------------------------------------------------------

    "nsworkspace_reference.txt": {
        "url":
            "https://developer.apple.com/documentation/appkit/nsworkspace",

        "topic": "nsworkspace",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    # ---------------------------------------------------------
    # ScreenCaptureKit
    # ---------------------------------------------------------

    "screencapturekit_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/screencapturekit",

        "topic": "screencapturekit",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "screencapturekit_macos_capture.txt": {
        "url":
            "https://developer.apple.com/documentation/screencapturekit/"
            "capturing-screen-content-in-macos",

        "topic": "screencapturekit",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    # ---------------------------------------------------------
    # Pasteboard
    # ---------------------------------------------------------

    "nspasteboard_reference.txt": {
        "url":
            "https://developer.apple.com/documentation/appkit/nspasteboard",

        "topic": "pasteboard",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "pasteboard_concepts.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "Cocoa/Conceptual/PasteboardGuide106/Articles/pbConcepts.html",

        "topic": "pasteboard",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },

    # ---------------------------------------------------------
    # Windows / AppKit
    # ---------------------------------------------------------

    "nswindow_reference.txt": {
        "url":
            "https://developer.apple.com/documentation/appkit/nswindow",

        "topic": "appkit_windows",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "windows_panels_screens.txt": {
        "url":
            "https://developer.apple.com/documentation/appkit/"
            "windows-panels-and-screens",

        "topic": "appkit_windows",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    # ---------------------------------------------------------
    # Notifications
    # ---------------------------------------------------------

    "usernotifications_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/usernotifications",

        "topic": "notifications",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "scheduling_local_notifications.txt": {
        "url":
            "https://developer.apple.com/documentation/usernotifications/"
            "scheduling-a-notification-locally-from-your-app",

        "topic": "notifications",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    # ---------------------------------------------------------
    # Filesystem / FSEvents
    # ---------------------------------------------------------

    "fsevents_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/coreservices/"
            "file_system_events",

        "topic": "filesystem",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "fsevents_programming_guide.txt": {
        "url":
            "https://developer.apple.com/library/archive/documentation/"
            "Darwin/Conceptual/FSEvents_ProgGuide/"
            "UsingtheFSEventsFramework/"
            "UsingtheFSEventsFramework.html",

        "topic": "filesystem",
        "publisher": "Apple",
        "authority": "official_archive",
        "authority_score": 0.90,
    },

    # ---------------------------------------------------------
    # Sandbox / TCC / Permissions
    # ---------------------------------------------------------

    "app_sandbox_overview.txt": {
        "url":
            "https://developer.apple.com/documentation/security/app-sandbox",

        "topic": "permissions",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "configuring_macos_app_sandbox.txt": {
        "url":
            "https://developer.apple.com/documentation/xcode/"
            "configuring-the-macos-app-sandbox",

        "topic": "permissions",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "apple_events_usage_description.txt": {
        "url":
            "https://developer.apple.com/documentation/bundleresources/"
            "information_property_list/"
            "nsappleeventsusagedescription",

        "topic": "permissions",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },

    "info_plist_privacy_keys.txt": {
        "url":
            "https://developer.apple.com/documentation/bundleresources/"
            "information_property_list/"
            "managing-your-app-s-information-property-list",

        "topic": "permissions",
        "publisher": "Apple",
        "authority": "official",
        "authority_score": 1.0,
    },
}

def fetch_text(url: str) -> str | None:
    """
    Fetch readable documentation text.

    Strategy:
    1. For modern Apple Developer documentation, try the
       official Markdown representation first.
    2. Otherwise fall back to Trafilatura HTML extraction.
    """

    # Modern Apple Developer docs expose Markdown versions.
    if (
        url.startswith("https://developer.apple.com/documentation/")
        and not url.endswith(".md")
    ):
        markdown_url = url.rstrip("/") + ".md"

        try:
            response = requests.get(
                markdown_url,
                timeout=20,
                headers={
                    "User-Agent":
                        "Mozilla/5.0 StellaMacKnowledge/1.0"
                },
            )

            if response.ok:
                text = response.text.strip()

                # Avoid accepting tiny/error responses.
                if len(text) >= 500:
                    return text

        except requests.RequestException:
            pass

    # Normal HTML / archived Apple docs / Apple Support /
    # community sources.
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        return None

    text = trafilatura.extract(
        downloaded,
        include_links=False,
        include_tables=True,
        include_comments=False,
        favor_precision=True,
    )

    if not text:
        return None

    text = text.strip()

    # Reject JavaScript-shell placeholder pages.
    if (
        len(text) < 500
        and "requires JavaScript" in text
    ):
        return None

    return text

def build_header(metadata: dict) -> str:
    return (
        f"Source-URL: {metadata['url']}\n"
        f"Publisher: {metadata['publisher']}\n"
        f"Topic: {metadata['topic']}\n"
        f"Authority: {metadata['authority']}\n"
        f"Authority-Score: {metadata['authority_score']}\n"
    )


def fetch_one(name: str, metadata: dict) -> bool:
    url = metadata["url"]

    print(f"Fetching {name}")
    print(f"  {url}")

    text = fetch_text(url)

    MIN_DOCUMENT_CHARS = 500

    if not text or len(text.strip()) < MIN_DOCUMENT_CHARS:
        print(
            f"  FAILED: document too small "
            f"({len(text.strip()) if text else 0} chars)"
        )
        return False

    out_path = RAW_DIR / name

    output = (
        build_header(metadata)
        + "\n"
        + text
        + "\n"
    )

    out_path.write_text(
        output,
        encoding="utf-8",
    )

    print(
        f"  saved {len(text):,} chars "
        f"[{metadata['authority']}]"
    )

    return True


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = 0

    for name, metadata in SOURCES.items():
        if fetch_one(name, metadata):
            successful += 1

    print()
    print(
        f"Done: {successful}/{len(SOURCES)} "
        f"documents saved to {RAW_DIR}/"
    )

    print()
    print("Next:")
    print("  python3 core/ingest_mac.py")
    print("  python3 core/chunk_mac.py")
    print("  python3 core/embed_mac.py")


if __name__ == "__main__":
    main()