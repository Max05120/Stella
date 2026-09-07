"""
Deterministic filesystem action parser.

Currently supports:

- create folder
- move file
- move files
- copy file
- copy files
- rename file
"""

from __future__ import annotations

import re
from pathlib import Path

from core.actions.models import ActionRequest
from core.actions.parsers.common import (
    clean_path,
    looks_like_path,
    resolve_folder_location,
    split_file_names,
    strip_polite_prefixes,
    valid_file_name,
    valid_folder_name,
)


def plan_filesystem_action(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    """
    Try filesystem parsers in deterministic order.
    """

    planners = [
        _plan_trash_files,
        _plan_trash_file,
        _plan_create_folder,
        _plan_move_files,
        _plan_move_file,
        _plan_copy_files,
        _plan_copy_file,
        _plan_rename_file,
    ]

    for planner in planners:
        result = planner(
            raw_text,
            normalized,
        )

        if result:
            return result

    return None

def _plan_trash_files(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    """
    Understand explicit multi-file Trash requests.

    Supported examples:

        Trash report.pdf and notes.txt from Desktop

        Put report.pdf and notes.txt from Desktop in the Trash

        Move report.pdf and notes.txt from Desktop to Trash
    """

    text = strip_polite_prefixes(
        normalized
    )

    patterns = [
        (
            r"^trash\s+"
            r"(.+?)\s+"
            r"from\s+"
            r"(.+)$"
        ),
        (
            r"^put\s+"
            r"(.+?)\s+"
            r"from\s+"
            r"(.+?)\s+"
            r"in\s+(?:the\s+)?trash$"
        ),
        (
            r"^move\s+"
            r"(.+?)\s+"
            r"from\s+"
            r"(.+?)\s+"
            r"to\s+(?:the\s+)?trash$"
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

        file_names = split_file_names(
            match.group(1)
        )

        # This parser handles plural requests only.
        if len(file_names) < 2:
            return None

        source_directory = (
            resolve_folder_location(
                match.group(2)
            )
        )

        if source_directory is None:
            return None

        for file_name in file_names:
            if not valid_file_name(
                file_name
            ):
                return None

        source_paths = [
            str(
                Path(source_directory)
                / file_name
            )
            for file_name in file_names
        ]

        return ActionRequest(
            raw_text=raw_text,
            action="trash",
            target_type="files",
            target=", ".join(
                source_paths
            ),
            arguments={
                "source_paths": source_paths,
            },
        )

    return None

def _plan_trash_file(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    """
    Understand explicit single-file Trash requests.

    Supported examples:

        Trash report.pdf from Desktop
        Put report.pdf from Desktop in the Trash
        Move report.pdf from Desktop to Trash
        Trash ~/Desktop/report.pdf
    """

    text = strip_polite_prefixes(
        normalized
    )

    # ---------------------------------------------------------
    # Form 1:
    #
    # Trash report.pdf from Desktop
    # ---------------------------------------------------------

    match = re.match(
        r"^trash\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        file_name = (
            match.group(1)
            .strip()
        )

        location = (
            match.group(2)
            .strip()
        )

        if not valid_file_name(
            file_name
        ):
            return None

        source_directory = (
            resolve_folder_location(
                location
            )
        )

        if source_directory is None:
            return None

        source_path = str(
            Path(source_directory)
            / file_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="trash",
            target_type="file",
            target=source_path,
            arguments={
                "source_path": source_path,
            },
        )

    # ---------------------------------------------------------
    # Form 2:
    #
    # Put report.pdf from Desktop in the Trash
    # ---------------------------------------------------------

    match = re.match(
        r"^put\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+?)\s+"
        r"in\s+(?:the\s+)?trash$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        file_name = (
            match.group(1)
            .strip()
        )

        if not valid_file_name(
            file_name
        ):
            return None

        source_directory = (
            resolve_folder_location(
                match.group(2)
            )
        )

        if source_directory is None:
            return None

        source_path = str(
            Path(source_directory)
            / file_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="trash",
            target_type="file",
            target=source_path,
            arguments={
                "source_path": source_path,
            },
        )

    # ---------------------------------------------------------
    # Form 3:
    #
    # Move report.pdf from Desktop to Trash
    # ---------------------------------------------------------

    match = re.match(
        r"^move\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+?)\s+"
        r"to\s+(?:the\s+)?trash$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        file_name = (
            match.group(1)
            .strip()
        )

        if not valid_file_name(
            file_name
        ):
            return None

        source_directory = (
            resolve_folder_location(
                match.group(2)
            )
        )

        if source_directory is None:
            return None

        source_path = str(
            Path(source_directory)
            / file_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="trash",
            target_type="file",
            target=source_path,
            arguments={
                "source_path": source_path,
            },
        )

    # ---------------------------------------------------------
    # Form 4:
    #
    # Trash ~/Desktop/report.pdf
    # ---------------------------------------------------------

    match = re.match(
        r"^trash\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    source_path = clean_path(
        match.group(1)
    )

    if not looks_like_path(
        source_path
    ):
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="trash",
        target_type="file",
        target=source_path,
        arguments={
            "source_path": source_path,
        },
    )

def _plan_create_folder(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    patterns = [
        (
            r"^(?:create|make)\s+"
            r"(?:a\s+)?folder\s+"
            r"(?:called|named)\s+"
            r"(.+?)\s+"
            r"(?:on|in)\s+"
            r"(.+)$"
        ),
        (
            r"^(?:create|make)\s+"
            r"(?:a\s+)?folder\s+"
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

        folder_name = (
            match.group(1)
            .strip()
        )

        location = (
            match.group(2)
            .strip()
        )

        if not valid_folder_name(
            folder_name
        ):
            return None

        parent_path = (
            resolve_folder_location(
                location
            )
        )

        if parent_path is None:
            return None

        target_path = (
            Path(parent_path)
            / folder_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="create",
            target_type="folder",
            target=str(target_path),
            arguments={
                "parent_path": parent_path,
                "folder_name": folder_name,
            },
        )

    return None


def _plan_move_files(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^(?:move|relocate)\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+?)\s+"
        r"(?:to|into)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    file_names = split_file_names(
        match.group(1)
    )

    if len(file_names) < 2:
        return None

    source_directory = (
        resolve_folder_location(
            match.group(2)
        )
    )

    destination_directory = (
        resolve_folder_location(
            match.group(3)
        )
    )

    if source_directory is None:
        return None

    if destination_directory is None:
        return None

    for file_name in file_names:
        if not valid_file_name(
            file_name
        ):
            return None

    source_paths = [
        str(
            Path(source_directory)
            / file_name
        )
        for file_name in file_names
    ]

    return ActionRequest(
        raw_text=raw_text,
        action="move",
        target_type="files",
        target=", ".join(
            source_paths
        ),
        arguments={
            "source_paths": source_paths,
            "destination_path": destination_directory,
        },
    )


def _plan_move_file(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    # Natural location form:
    #
    # Move report.pdf from Desktop to Downloads

    match = re.match(
        r"^(?:move|relocate)\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+?)\s+"
        r"(?:to|into)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        file_name = (
            match.group(1)
            .strip()
        )

        if not valid_file_name(
            file_name
        ):
            return None

        source_directory = (
            resolve_folder_location(
                match.group(2)
            )
        )

        destination_directory = (
            resolve_folder_location(
                match.group(3)
            )
        )

        if source_directory is None:
            return None

        if destination_directory is None:
            return None

        source_path = str(
            Path(source_directory)
            / file_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="move",
            target_type="file",
            target=source_path,
            arguments={
                "source_path": source_path,
                "destination_path": destination_directory,
            },
        )

    # Explicit path form:
    #
    # Move ~/Desktop/report.pdf to Downloads

    match = re.match(
        r"^(?:move|relocate)\s+"
        r"(.+?)\s+"
        r"(?:to|into)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    source_path = clean_path(
        match.group(1)
    )

    if not looks_like_path(
        source_path
    ):
        return None

    destination_path = (
        resolve_folder_location(
            match.group(2)
        )
    )

    if destination_path is None:
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="move",
        target_type="file",
        target=source_path,
        arguments={
            "source_path": source_path,
            "destination_path": destination_path,
        },
    )


def _plan_copy_files(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    match = re.match(
        r"^(?:copy|duplicate)\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+?)\s+"
        r"(?:to|into)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    file_names = split_file_names(
        match.group(1)
    )

    if len(file_names) < 2:
        return None

    source_directory = (
        resolve_folder_location(
            match.group(2)
        )
    )

    destination_directory = (
        resolve_folder_location(
            match.group(3)
        )
    )

    if source_directory is None:
        return None

    if destination_directory is None:
        return None

    for file_name in file_names:
        if not valid_file_name(
            file_name
        ):
            return None

    source_paths = [
        str(
            Path(source_directory)
            / file_name
        )
        for file_name in file_names
    ]

    return ActionRequest(
        raw_text=raw_text,
        action="copy",
        target_type="files",
        target=", ".join(
            source_paths
        ),
        arguments={
            "source_paths": source_paths,
            "destination_path": destination_directory,
        },
    )


def _plan_copy_file(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    # Natural source-location form.

    match = re.match(
        r"^(?:copy|duplicate)\s+"
        r"(.+?)\s+"
        r"from\s+"
        r"(.+?)\s+"
        r"(?:to|into)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        file_name = (
            match.group(1)
            .strip()
        )

        if not valid_file_name(
            file_name
        ):
            return None

        source_directory = (
            resolve_folder_location(
                match.group(2)
            )
        )

        destination_directory = (
            resolve_folder_location(
                match.group(3)
            )
        )

        if source_directory is None:
            return None

        if destination_directory is None:
            return None

        source_path = str(
            Path(source_directory)
            / file_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="copy",
            target_type="file",
            target=source_path,
            arguments={
                "source_path": source_path,
                "destination_path": destination_directory,
            },
        )

    # Explicit source path form.

    match = re.match(
        r"^(?:copy|duplicate)\s+"
        r"(.+?)\s+"
        r"(?:to|into)\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    source_path = clean_path(
        match.group(1)
    )

    if not looks_like_path(
        source_path
    ):
        return None

    destination_path = (
        resolve_folder_location(
            match.group(2)
        )
    )

    if destination_path is None:
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="copy",
        target_type="file",
        target=source_path,
        arguments={
            "source_path": source_path,
            "destination_path": destination_path,
        },
    )


def _plan_rename_file(
    raw_text: str,
    normalized: str,
) -> ActionRequest | None:
    text = strip_polite_prefixes(
        normalized
    )

    # Natural location form:
    #
    # Rename report.txt on Desktop to final-report.txt

    match = re.match(
        r"^rename\s+"
        r"(.+?)\s+"
        r"(?:on|in)\s+"
        r"(.+?)\s+"
        r"to\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        file_name = (
            match.group(1)
            .strip()
        )

        location = (
            match.group(2)
            .strip()
        )

        new_name = (
            match.group(3)
            .strip()
        )

        if not valid_file_name(
            file_name
        ):
            return None

        if not valid_file_name(
            new_name
        ):
            return None

        source_directory = (
            resolve_folder_location(
                location
            )
        )

        if source_directory is None:
            return None

        source_path = str(
            Path(source_directory)
            / file_name
        )

        return ActionRequest(
            raw_text=raw_text,
            action="rename",
            target_type="file",
            target=source_path,
            arguments={
                "source_path": source_path,
                "new_name": new_name,
            },
        )

    # Explicit path form.

    match = re.match(
        r"^rename\s+"
        r"(.+?)\s+"
        r"to\s+"
        r"(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    source_path = clean_path(
        match.group(1)
    )

    new_name = (
        match.group(2)
        .strip()
    )

    if not looks_like_path(
        source_path
    ):
        return None

    if not valid_file_name(
        new_name
    ):
        return None

    return ActionRequest(
        raw_text=raw_text,
        action="rename",
        target_type="file",
        target=source_path,
        arguments={
            "source_path": source_path,
            "new_name": new_name,
        },
    )