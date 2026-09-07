"""
Shared helpers for deterministic action parsers.
"""

from __future__ import annotations

import re
from pathlib import Path


SPECIAL_FOLDERS = {
    "desktop": Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "movies": Path.home() / "Movies",
    "music": Path.home() / "Music",
    "pictures": Path.home() / "Pictures",
    "applications": Path("/Applications"),
}


def normalize(text: str) -> str:
    text = text.strip()

    # Collapse repeated whitespace without aggressively
    # changing paths or URLs.
    return re.sub(
        r"\s+",
        " ",
        text,
    )


def strip_polite_prefixes(
    text: str,
) -> str:
    """
    Remove lightweight conversational prefixes.

    This intentionally does not perform general NLP.
    """

    patterns = [
        r"^please\s+",
        r"^can you\s+",
        r"^could you\s+",
        r"^would you\s+",
        r"^stella[,\s]+",
        r"^hey stella[,\s]+",
    ]

    result = text.strip()

    changed = True

    while changed:
        changed = False

        for pattern in patterns:
            updated = re.sub(
                pattern,
                "",
                result,
                flags=re.IGNORECASE,
            )

            if updated != result:
                result = updated.strip()
                changed = True

    return result


def clean_path(
    value: str,
) -> str:
    value = value.strip()

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]

    return value.strip()


def looks_like_path(
    value: str,
) -> bool:
    if not value:
        return False

    return (
        value.startswith("/")
        or value.startswith("~/")
        or value.startswith("./")
        or value.startswith("../")
    )


def resolve_folder_location(
    value: str,
) -> str | None:
    """
    Convert things such as:

        Desktop
        my Desktop
        Downloads folder
        ~/Documents
        /Users/max/Desktop

    into a filesystem parent path.
    """

    value = value.strip()

    if not value:
        return None

    cleaned = re.sub(
        r"^my\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s+folder$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    special = SPECIAL_FOLDERS.get(
        cleaned.lower()
    )

    if special is not None:
        return str(special)

    path = clean_path(cleaned)

    if looks_like_path(path):
        return path

    return None


def valid_file_name(
    value: str,
) -> bool:
    name = value.strip()

    if not name:
        return False

    if name in {".", ".."}:
        return False

    if "/" in name:
        return False

    return True


def valid_folder_name(
    value: str,
) -> bool:
    name = value.strip()

    if not name:
        return False

    if name in {".", ".."}:
        return False

    if "/" in name:
        return False

    return True


def split_file_names(
    value: str,
) -> list[str]:
    """
    Split conservative lists such as:

        report.pdf and notes.txt

        report.pdf, notes.txt and data.csv
    """

    value = value.strip()

    if not value:
        return []

    value = re.sub(
        r"\s*,\s*and\s+",
        ", ",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s+and\s+",
        ", ",
        value,
        flags=re.IGNORECASE,
    )

    parts = [
        part.strip()
        for part in value.split(",")
    ]

    return [
        part
        for part in parts
        if part
    ]