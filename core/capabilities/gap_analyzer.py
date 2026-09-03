"""
gap_analyzer.py

Analyzes missing capabilities using Stella's macOS knowledge base.

This module does NOT execute anything.

It classifies a missing capability as:

IMPLEMENTABLE
    Strong enough evidence exists that macOS supports the action.

BLOCKED
    The capability appears technically possible but requires
    permission/platform conditions that are currently unavailable.

UNKNOWN
    Stella does not have enough relevant evidence to confidently
    describe an implementation.
"""

from dataclasses import dataclass, field
from typing import Any

from core.mac_retriever import retrieve_mac


@dataclass
class CapabilityGap:
    query: str

    status: str
    # implementable | blocked | unknown

    confidence: float = 0.0

    suggested_name: str | None = None
    suggested_family: str | None = None

    likely_implementation: list[str] = field(
        default_factory=list
    )

    required_permissions: list[str] = field(
        default_factory=list
    )

    risk: str = "unknown"

    explanation: str = ""

    evidence: list[dict[str, Any]] = field(
        default_factory=list
    )


FAMILY_HINTS = {
    "file": "filesystem",
    "files": "filesystem",
    "folder": "filesystem",
    "move": "filesystem",
    "copy": "filesystem",
    "rename": "filesystem",
    "delete": "filesystem",
    "trash": "filesystem",

    "finder": "finder",

    "window": "windows",
    "windows": "windows",

    "click": "accessibility",
    "button": "accessibility",
    "menu": "accessibility",
    "type": "accessibility",
    "checkbox": "accessibility",

    "screen": "screen_capture",
    "capture": "screen_capture",
    "screenshot": "screen_capture",

    "shortcut": "shortcuts",

    "calendar": "calendar",
    "event": "calendar",

    "reminder": "reminders",

    "clipboard": "clipboard",

    "notification": "notifications",

    "launch agent": "background_tasks",
    "background task": "background_tasks",

    "watch folder": "filesystem_monitoring",
    "monitor folder": "filesystem_monitoring",
}


TOPIC_FAMILY_MAP = {
    "filesystem": {
        "filesystem",
        "permissions",
        "nsworkspace",
        "applescript",
    },

    "finder": {
        "filesystem",
        "applescript",
        "accessibility",
        "nsworkspace",
    },

    "windows": {
        "appkit_windows",
        "accessibility",
    },

    "accessibility": {
        "accessibility",
    },

    "screen_capture": {
        "screencapturekit",
        "permissions",
    },

    "shortcuts": {
        "shortcuts",
    },

    "calendar": {
        "eventkit",
        "permissions",
    },

    "reminders": {
        "eventkit",
        "permissions",
    },

    "clipboard": {
        "pasteboard",
    },

    "notifications": {
        "notifications",
        "permissions",
    },

    "background_tasks": {
        "launchd",
    },

    "filesystem_monitoring": {
        "filesystem",
        "launchd",
    },
}

ACTION_FAMILY_HINTS = {
    "delete": "filesystem",
    "trash": "filesystem",
    "erase": "filesystem",
    "move": "filesystem",
    "copy": "filesystem",
    "rename": "filesystem",

    "click": "accessibility",
    "press": "accessibility",
    "type": "accessibility",

    "capture": "screen_capture",
    "record": "screen_capture",

    "watch": "filesystem_monitoring",
    "monitor": "filesystem_monitoring",

    "run": None,
}

FAMILY_RETRIEVAL_QUERIES = {
    "filesystem": (
        "macOS FileManager move copy rename remove delete "
        "trash files directories filesystem operations"
    ),

    "finder": (
        "macOS Finder automation AppleScript files folders "
        "selection tags"
    ),

    "windows": (
        "macOS window management Accessibility AppKit "
        "move resize focus windows"
    ),

    "accessibility": (
        "macOS Accessibility API AXUIElement interact with "
        "buttons menus text fields UI elements other applications"
    ),

    "screen_capture": (
        "macOS ScreenCaptureKit capture screen displays "
        "applications windows"
    ),

    "shortcuts": (
        "macOS Shortcuts command line run shortcut URL scheme"
    ),

    "calendar": (
        "macOS EventKit create update retrieve calendar events"
    ),

    "reminders": (
        "macOS EventKit reminders create update retrieve"
    ),

    "clipboard": (
        "macOS NSPasteboard clipboard read write"
    ),

    "notifications": (
        "macOS UserNotifications local notifications"
    ),

    "background_tasks": (
        "macOS launchd LaunchAgent background task"
    ),

    "filesystem_monitoring": (
        "macOS FSEvents watch directory filesystem changes "
        "new files"
    ),
}

IMPLEMENTATION_HINTS = {
    "filesystem": [
        "FileManager",
        "Finder scripting",
    ],

    "finder": [
        "Finder AppleScript",
        "Accessibility APIs",
    ],

    "windows": [
        "Accessibility APIs",
        "AppKit",
    ],

    "accessibility": [
        "AXUIElement",
        "Accessibility APIs",
    ],

    "screen_capture": [
        "ScreenCaptureKit",
    ],

    "shortcuts": [
        "Shortcuts CLI",
        "Shortcuts URL scheme",
    ],

    "calendar": [
        "EventKit",
    ],

    "reminders": [
        "EventKit",
    ],

    "clipboard": [
        "NSPasteboard",
    ],

    "notifications": [
        "UserNotifications",
    ],

    "background_tasks": [
        "launchd",
        "LaunchAgents",
    ],

    "filesystem_monitoring": [
        "FSEvents",
    ],
}


PERMISSION_HINTS = {
    "filesystem": [
        "filesystem access",
    ],

    "finder": [
        "Automation / Apple Events",
    ],

    "windows": [
        "Accessibility",
    ],

    "accessibility": [
        "Accessibility",
    ],

    "screen_capture": [
        "Screen Recording",
    ],

    "calendar": [
        "Calendar",
    ],

    "reminders": [
        "Reminders",
    ],

    "clipboard": [],

    "notifications": [
        "Notifications",
    ],

    "background_tasks": [],

    "filesystem_monitoring": [
        "filesystem access",
    ],
}


DESTRUCTIVE_TERMS = {
    "delete",
    "erase",
    "remove",
    "trash",
    "overwrite",
    "empty trash",
    "force quit",
}


def _normalize(text: str) -> str:
    return " ".join(
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .split()
    )


def _guess_family(
    query: str,
) -> str | None:

    normalized = _normalize(query)

    words = normalized.split()

    # --------------------------------------------------
    # 1. Strong action-based inference
    # --------------------------------------------------

    for word in words:

        if word in ACTION_FAMILY_HINTS:

            family = ACTION_FAMILY_HINTS[word]

            if family:
                return family

    # --------------------------------------------------
    # 2. Special compound intents
    # --------------------------------------------------

    if (
        "shortcut" in words
        and (
            "run" in words
            or "execute" in words
        )
    ):
        return "shortcuts"

    if (
        "calendar" in words
        or "event" in words
    ):
        return "calendar"

    if "reminder" in words:
        return "reminders"

    # --------------------------------------------------
    # 3. Object/topic fallback
    # --------------------------------------------------

    ordered_hints = sorted(
        FAMILY_HINTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for phrase, family in ordered_hints:

        if phrase in normalized:
            return family

    return None


def _guess_risk(
    query: str,
) -> str:

    normalized = _normalize(query)

    for term in DESTRUCTIVE_TERMS:
        if term in normalized:
            return "destructive"

    return "low"


def _suggest_capability_name(
    query: str,
    family: str | None,
) -> str:

    normalized = (
        _normalize(query)
        .replace(" ", "_")
    )

    if family:
        return f"{family}_{normalized}"

    return normalized


def _topic_matches_family(
    topic: str | None,
    family: str | None,
) -> bool:

    if not topic or not family:
        return False

    allowed_topics = TOPIC_FAMILY_MAP.get(
        family,
        set(),
    )

    return topic in allowed_topics


def _distance_quality(
    distance: float | None,
) -> float:
    """
    Convert Chroma distance into a rough 0-1 relevance score.

    The exact scale depends on the embedding setup, so this is
    intentionally conservative.

    Based on our current retrieval tests:
        ~190-220 = strong
        ~250-300 = usable
        >350     = weak
    """

    if distance is None:
        return 0.0

    if distance <= 220:
        return 1.0

    if distance <= 260:
        return 0.85

    if distance <= 300:
        return 0.65

    if distance <= 340:
        return 0.40

    if distance <= 380:
        return 0.20

    return 0.0


def _evidence_score(
    result: dict,
    family: str | None,
) -> float:

    distance_score = _distance_quality(
        result.get("distance")
    )

    authority_score = float(
        result.get(
            "authority_score",
            0.5,
        )
        or 0.5
    )

    topic_score = (
        1.0
        if _topic_matches_family(
            result.get("topic"),
            family,
        )
        else 0.0
    )

    # Relevance matters most.
    return (
        distance_score * 0.60
        + topic_score * 0.25
        + authority_score * 0.15
    )


def _analyze_evidence(
    results: list[dict],
    family: str | None,
) -> tuple[float, list[dict]]:

    scored = []

    for result in results:

        score = _evidence_score(
            result,
            family,
        )

        scored.append({
            **result,
            "evidence_score": score,
        })

    scored.sort(
        key=lambda item:
            item["evidence_score"],
        reverse=True,
    )

    if not scored:
        return 0.0, []

    top_scores = [
        item["evidence_score"]
        for item in scored[:3]
    ]

    confidence = (
        sum(top_scores)
        / len(top_scores)
    )

    return confidence, scored


def analyze_capability_gap(
    query: str,
    top_k: int = 6,
) -> CapabilityGap:

    family = _guess_family(query)

    suggested_name = (
        _suggest_capability_name(
            query,
            family,
        )
    )

    retrieval_query = (
        FAMILY_RETRIEVAL_QUERIES.get(
            family,
            query,
        )
        if family
        else query
    )

    results = retrieve_mac(
        retrieval_query,
        top_k=top_k,
    )

    if not results:

        return CapabilityGap(
            query=query,
            status="unknown",
            confidence=0.0,
            suggested_name=suggested_name,
            suggested_family=family,
            risk=_guess_risk(query),
            explanation=(
                "Stella does not currently have an "
                "executable capability for this action, "
                "and no relevant macOS implementation "
                "evidence was found."
            ),
        )

    confidence, scored_results = (
        _analyze_evidence(
            results,
            family,
        )
    )

    evidence = []

    for result in scored_results:

        evidence.append({
            "source":
                result.get("source"),

            "topic":
                result.get("topic"),

            "authority":
                result.get("authority"),

            "authority_score":
                result.get(
                    "authority_score"
                ),

            "distance":
                result.get("distance"),

            "evidence_score":
                result.get(
                    "evidence_score"
                ),

            "text":
                result.get("text"),
        })

    implementations = (
        IMPLEMENTATION_HINTS.get(
            family,
            [],
        )
        if family
        else []
    )

    permissions = (
        PERMISSION_HINTS.get(
            family,
            [],
        )
        if family
        else []
    )

    matching_topic_count = sum(
        1
        for result in scored_results[:4]
        if _topic_matches_family(
            result.get("topic"),
            family,
        )
    )

    # Strong enough to claim implementability.
    if (
        family
        and confidence >= 0.62
        and matching_topic_count >= 1
    ):

        return CapabilityGap(
            query=query,
            status="implementable",
            confidence=confidence,
            suggested_name=suggested_name,
            suggested_family=family,
            likely_implementation=implementations,
            required_permissions=permissions,
            risk=_guess_risk(query),
            explanation=(
                "Stella does not currently have an "
                "executable capability for this action, "
                "but the macOS knowledge base contains "
                "relevant implementation evidence."
            ),
            evidence=evidence,
        )

    return CapabilityGap(
        query=query,
        status="unknown",
        confidence=confidence,
        suggested_name=suggested_name,
        suggested_family=family,
        likely_implementation=[],
        required_permissions=[],
        risk=_guess_risk(query),
        explanation=(
            "Stella does not currently have an executable "
            "capability for this action. Some related macOS "
            "documentation was found, but the evidence is "
            "not strong enough to confidently claim that "
            "the requested capability is implementable."
        ),
        evidence=evidence,
    )