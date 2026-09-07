"""
Deterministic read-only filesystem discovery parser.

Currently supports:

- list files in a common folder
- list files by extension
"""

from __future__ import annotations

import re

from core.actions.models import ActionRequest

from core.actions.parsers.common import (
    resolve_folder_location,
    strip_polite_prefixes,
)


FILE_TYPE_EXTENSIONS = {
    "pdf": ".pdf",
    "pdfs": ".pdf",

    "text file": ".txt",
    "text files": ".txt",
    "txt": ".txt",

    "image": None,
    "images": None,

    "png": ".png",
    "pngs": ".png",

    "jpg": ".jpg",
    "jpgs": ".jpg",
    "jpeg": ".jpeg",
    "jpegs": ".jpeg",

    "zip": ".zip",
    "zips": ".zip",
}


def plan_discovery_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:

    result = _plan_list_typed_files(
        raw_text,
        normalized,
    )

    if result:
        return result

    result = _plan_list_files(
        raw_text,
        normalized,
    )

    if result:
        return result

    return None


def _plan_list_typed_files(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    """
    Examples:

        Show me the PDFs on my Desktop
        List PDFs in Downloads
        Find PNGs on Desktop
    """

    text = strip_polite_prefixes(
        normalized
    )

    patterns = [
        (
            r"^(?:show|list|find)"
            r"(?:\s+me)?\s+"
            r"(?:the\s+)?"
            r"(.+?)\s+"
            r"(?:on|in)\s+"
            r"(.+)$"
        ),
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        file_type = (
            match.group(1)
            .strip()
            .lower()
        )

        location = (
            match.group(2)
            .strip()
        )

        if file_type not in FILE_TYPE_EXTENSIONS:
            return None

        extension = FILE_TYPE_EXTENSIONS[
            file_type
        ]

        # Don't claim support for broad image discovery yet.
        if extension is None:
            return None

        directory_path = (
            resolve_folder_location(
                location
            )
        )

        if directory_path is None:
            return None

        return ActionRequest(
            raw_text=raw_text,
            action="list",
            target_type="files",
            target=directory_path,
            arguments={
                "directory_path": directory_path,
                "extension": extension,
            },
        )

    return None


def _plan_list_files(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    """
    Examples:

        Show me the files on my Desktop
        List files in Downloads
        Show files in Documents
    """

    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^(?:show|list)"
        r"(?:\s+me)?\s+"
        r"(?:the\s+)?files\s+"
        r"(?:on|in)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    directory_path = (
        resolve_folder_location(
            match.group(1)
        )
    )

    if directory_path is None:
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="list",
        target_type="files",
        target=directory_path,
        arguments={
            "directory_path": directory_path,
            "extension": None,
        },
    )