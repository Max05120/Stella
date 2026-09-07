"""
Deterministic user-facing responses for executed Mac actions.

StellaEngine should not need to know how every capability
formats its success or failure message.
"""


def format_action_success(
    result,
) -> str:
    request = result.request

    if request is None:
        return result.message

    # ---------------------------------------------------------
    # Workspace
    # ---------------------------------------------------------

    if (
        request.action == "open"
        and request.target_type == "application"
    ):
        return (
            f"Opened {request.target}."
        )

    if (
        request.action == "open"
        and request.target_type == "url"
    ):
        return (
            f"Opened {request.target}."
        )

    if (
        request.action == "open"
        and request.target_type == "path"
    ):
        return (
            f"Opened {request.target}."
        )

    if (
        request.action == "reveal"
        and request.target_type == "path"
    ):
        return (
            f"Revealed {request.target} in Finder."
        )
    
    # ---------------------------------------------------------
    # Trash 
    # ---------------------------------------------------------
    if (
        request.action == "trash"
        and request.target_type == "files"
    ):
        data = _execution_data(
            result
        )

        count = data.get(
            "count"
        )

        if count is not None:
            return (
                f"Moved {count} files to the Trash."
            )

        return (
            "Moved the files to the Trash."
        )
    
    if (
        request.action == "trash"
        and request.target_type == "file"
    ):
        data = _execution_data(
            result
        )

        file_name = data.get(
            "file_name"
        )

        if file_name:
            return (
                f"Moved {file_name} to the Trash."
            )

        return "Moved the file to the Trash."

    # ---------------------------------------------------------
    # Create folder
    # ---------------------------------------------------------

    if (
        request.action == "create"
        and request.target_type == "folder"
    ):
        return (
            f"Created folder {request.target}."
        )

    # ---------------------------------------------------------
    # Move
    # ---------------------------------------------------------

    if (
        request.action == "move"
        and request.target_type == "file"
    ):
        data = _execution_data(
            result
        )

        target_path = data.get(
            "target_path"
        )

        if target_path:
            return (
                f"Moved file to {target_path}."
            )

        return (
            f"Moved {request.target}."
        )

    if (
        request.action == "move"
        and request.target_type == "files"
    ):
        data = _execution_data(
            result
        )

        count = data.get(
            "count"
        )

        destination = data.get(
            "destination_path"
        )

        if (
            count is not None
            and destination
        ):
            return (
                f"Moved {count} files to "
                f"{destination}."
            )

        return "Moved the files."

    # ---------------------------------------------------------
    # Copy
    # ---------------------------------------------------------

    if (
        request.action == "copy"
        and request.target_type == "file"
    ):
        data = _execution_data(
            result
        )

        target_path = data.get(
            "target_path"
        )

        if target_path:
            return (
                f"Copied file to {target_path}."
            )

        return (
            f"Copied {request.target}."
        )

    if (
        request.action == "copy"
        and request.target_type == "files"
    ):
        data = _execution_data(
            result
        )

        count = data.get(
            "count"
        )

        destination = data.get(
            "destination_path"
        )

        if (
            count is not None
            and destination
        ):
            return (
                f"Copied {count} files to "
                f"{destination}."
            )

        return "Copied the files."

    # ---------------------------------------------------------
    # Rename
    # ---------------------------------------------------------

    if (
        request.action == "rename"
        and request.target_type == "file"
    ):
        data = _execution_data(
            result
        )

        old_name = data.get(
            "old_name"
        )

        new_name = data.get(
            "new_name"
        )

        if old_name and new_name:
            return (
                f"Renamed {old_name} to {new_name}."
            )

        return (
            f"Renamed {request.target}."
        )
    # ---------------------------------------------------------
    # File discovery
    # ---------------------------------------------------------

    if (
        request.action == "list"
        and request.target_type == "files"
    ):
        data = _execution_data(result)

        file_names = data.get(
            "file_names",
            [],
        )

        count = data.get(
            "count",
            len(file_names),
        )

        directory = data.get(
            "directory_path",
        )

        if count == 0:
            return (
                "I couldn't find any matching files"
                + (
                    f" in {directory}."
                    if directory
                    else "."
                )
            )

        if count == 1:
            return (
                f"I found 1 file: {file_names[0]}"
            )

        preview_limit = 8
        preview = file_names[:preview_limit]

        names = ", ".join(preview)

        if count <= preview_limit:
            return (
                f"I found {count} files: {names}."
            )

        remaining = count - preview_limit

        return (
            f"I found {count} files. "
            f"Here are the first {preview_limit}: "
            f"{names}, plus {remaining} more."
        )
    
# ---------------------------------------------------------
# Finder context
# ---------------------------------------------------------

    if (
        request.action == "inspect"
        and request.target_type == "finder_selection"
    ):
        data = _execution_data(
            result
        )

        paths = data.get(
            "paths",
            []
        )

        count = len(paths)

        if count == 0:
            return (
                "You don't currently have anything "
                "selected in Finder."
            )

        if count == 1:
            return (
                f"You have this selected: "
                f"{paths[0]}"
            )

        names = [
            path.rstrip("/").split("/")[-1]
            for path in paths
        ]

        if count <= 8:
            return (
                f"You have {count} items selected: "
                + ", ".join(names)
                + "."
            )

        preview = ", ".join(
            names[:8]
        )

        return (
            f"You have {count} items selected. "
            f"The first 8 are: {preview}."
        )
    # ---------------------------------------------------------
    # Finder current location
    # ---------------------------------------------------------

    if (
        request.action == "inspect"
        and request.target_type == "finder_window_path"
    ):
        data = _execution_data(
            result
        )

        path = data.get(
            "path"
        )

        if not path:
            return (
                "There isn't an open Finder window."
            )

        return (
            f"The current Finder location is {path}."
        )
    # ---------------------------------------------------------
    # Finder windows
    # ---------------------------------------------------------

    if (
        request.action == "inspect"
        and request.target_type == "finder_windows"
    ):
        data = _execution_data(
            result
        )

        paths = data.get(
            "paths",
            []
        )

        count = len(paths)

        if count == 0:
            return (
                "There aren't any open Finder windows."
            )

        if count == 1:
            return (
                f"You have one Finder window open at "
                f"{paths[0]}."
            )

        return (
            f"You have {count} Finder windows open: "
            + ", ".join(paths[:8])
            + (
                "."
                if count <= 8
                else f", plus {count - 8} more."
            )
        )
    # Frontmost app

    if (
        request.action == "inspect"
        and request.target_type == "frontmost_application"
    ):
        data = _execution_data(result)

        name = data.get("name")

        if not name:
            return (
                "I couldn't determine the frontmost application."
            )

        return (
            f"The frontmost application is {name}."
        )


    # Clipboard read

    if (
        request.action == "inspect"
        and request.target_type == "clipboard"
    ):
        data = _execution_data(result)

        text = data.get("text")

        if not text:
            return (
                "The clipboard doesn't currently contain text."
            )

        return (
            f"Your clipboard contains: {text}"
        )


    # Clipboard write

    if (
        request.action == "write"
        and request.target_type == "clipboard"
    ):
        return (
            "Copied that to the clipboard."
        )


    # Volume read

    if (
        request.action == "inspect"
        and request.target_type == "system_volume"
    ):
        data = _execution_data(result)

        volume = data.get("volume")
        muted = data.get("muted")

        if volume is None:
            return (
                "I couldn't determine the current volume."
            )

        if muted:
            return (
                f"The volume is {volume}%, and the output is muted."
            )

        return (
            f"The volume is {volume}%."
        )


    # Volume set

    if (
        request.action == "set"
        and request.target_type == "system_volume"
    ):
        data = _execution_data(result)

        volume = data.get(
            "volume",
            request.target,
        )

        return (
            f"Set the volume to {volume}%."
        )


    if (
        request.action == "set"
        and request.target_type == "system_mute"
    ):
        return "Muted the system audio."


    if (
        request.action == "set"
        and request.target_type == "system_unmute"
    ):
        return "Unmuted the system audio."
    return result.message


def format_action_failure(
    result,
) -> str:
    request = result.request

    if request is None:
        return result.message

    # ---------------------------------------------------------
    # Workspace
    # ---------------------------------------------------------

    if (
        request.action == "open"
        and request.target_type == "application"
    ):
        return (
            f"I couldn't open {request.target}. "
            f"I couldn't find an installed application "
            f"with that name."
        )

    if (
        request.action == "open"
        and request.target_type == "path"
    ):
        return (
            f"I couldn't open {request.target}. "
            f"The path may not exist."
        )
    
    # ---------------------------------------------------------
    # Trash
    # ---------------------------------------------------------
    if (
        request.action == "trash"
        and request.target_type == "files"
    ):
        return (
            "I couldn't move those files to the Trash. "
            "One or more files may no longer exist "
            "or macOS may have blocked access."
        )
    
    if (
        request.action == "trash"
        and request.target_type == "file"
    ):
        return (
            f"I couldn't move {request.target} "
            f"to the Trash. The file may no longer "
            f"exist or macOS may have blocked access."
        )
    
    # ---------------------------------------------------------
    # Create
    # ---------------------------------------------------------

    if (
        request.action == "create"
        and request.target_type == "folder"
    ):
        return (
            f"I couldn't create {request.target}. "
            f"The destination may not exist, "
            f"may not be writable, or an item with "
            f"that name may already exist."
        )

    # ---------------------------------------------------------
    # Move
    # ---------------------------------------------------------

    if (
        request.action == "move"
        and request.target_type == "file"
    ):
        return (
            f"I couldn't move {request.target}. "
            f"The source may not exist, the destination "
            f"may be unavailable, or a file with the same "
            f"name may already exist there."
        )

    if (
        request.action == "move"
        and request.target_type == "files"
    ):
        return (
            "I couldn't move those files. "
            "One or more source files may not exist, "
            "the destination may be unavailable, "
            "or a file with the same name may already "
            "exist there."
        )

    # ---------------------------------------------------------
    # Copy
    # ---------------------------------------------------------

    if (
        request.action == "copy"
        and request.target_type == "file"
    ):
        return (
            f"I couldn't copy {request.target}. "
            f"The source may not exist, the destination "
            f"may be unavailable, or a file with the same "
            f"name may already exist there."
        )

    if (
        request.action == "copy"
        and request.target_type == "files"
    ):
        return (
            "I couldn't copy those files. "
            "One or more source files may not exist, "
            "the destination may be unavailable, "
            "or matching files may already exist there."
        )

    # ---------------------------------------------------------
    # Rename
    # ---------------------------------------------------------

    if (
        request.action == "rename"
        and request.target_type == "file"
    ):
        return (
            f"I couldn't rename {request.target}. "
            f"The file may not exist, the new name may "
            f"be invalid, or an item with that name may "
            f"already exist."
        )

    if (
        request.action == "list"
        and request.target_type == "files"
    ):
        return (
            f"I couldn't inspect {request.target}. "
            f"The folder may not exist or Stella may "
            f"not have permission to access it."
        )
    if (
        request.action == "inspect"
        and request.target_type == "finder_selection"
    ):
        return (
            "I couldn't read the current Finder selection. "
            "macOS may not have granted Stella permission "
            "to control Finder."
        )
    # ---------------------------------------------------------
    # Finder selection
    # ---------------------------------------------------------

    if (
        request.action == "inspect"
        and request.target_type == "finder_selection"
    ):
        data = _execution_data(result)

        paths = data.get(
            "paths",
            [],
        )

        count = len(paths)

        if count == 0:
            return (
                "Nothing is currently selected in Finder."
            )

        names = [
            path.rstrip("/").split("/")[-1]
            for path in paths
        ]

        if count == 1:
            return (
                f"You have {names[0]} selected."
            )

        return (
            f"You have {count} items selected: "
            + ", ".join(names[:10])
            + (
                "."
                if count <= 10
                else f", plus {count - 10} more."
            )
        )


    # ---------------------------------------------------------
    # Finder current location
    # ---------------------------------------------------------

    if (
        request.action == "inspect"
        and request.target_type == "finder_window_path"
    ):
        data = _execution_data(result)

        path = data.get(
            "path"
        )

        if not path:
            return (
                "There isn't an open Finder window."
            )

        return (
            f"The current Finder location is {path}."
        )


    # ---------------------------------------------------------
    # Finder windows
    # ---------------------------------------------------------

    if (
        request.action == "inspect"
        and request.target_type == "finder_windows"
    ):
        data = _execution_data(result)

        paths = data.get(
            "paths",
            [],
        )

        count = len(paths)

        if count == 0:
            return (
                "There aren't any open Finder windows."
            )

        if count == 1:
            return (
                f"You have one Finder window open at "
                f"{paths[0]}."
            )

        return (
            f"You have {count} Finder windows open: "
            + ", ".join(paths[:8])
            + (
                "."
                if count <= 8
                else f", plus {count - 8} more."
            )
        )


    # ---------------------------------------------------------
    # Finder navigation
    # ---------------------------------------------------------

    if (
        request.action == "open"
        and request.target_type == "finder_location"
    ):
        return (
            f"Opened {request.target} in Finder."
        )
    return result.message

def format_action_confirmation(
    result,
) -> str:
    request = result.request

    if request is None:
        return (
            "This action requires confirmation "
            "before I can execute it."
        )
    
    if (
        request.action == "trash"
        and request.target_type == "files"
    ):
        source_paths = (
            request.arguments.get(
                "source_paths",
                [],
            )
        )

        count = len(
            source_paths
        )

        return (
            f"Move {count} files to the Trash?"
        )

    if (
        request.action == "trash"
        and request.target_type == "file"
    ):
        file_name = (
            request.target
            .rstrip("/")
            .split("/")[-1]
        )

        return (
            f"Move {file_name} to the Trash?"
        )
    
    if request.target_type in {
        "finder_selection",
        "finder_window_path",
        "finder_windows",
    }:
        return (
            "I couldn't read Finder's current state. "
            "Finder automation access may be unavailable."
        )


    if (
        request.action == "open"
        and request.target_type == "finder_location"
    ):
        return (
            f"I couldn't open {request.target} in Finder."
        )

    return (
        "This action requires confirmation "
        "before I can execute it."
    )



def format_action_cancelled(
    result,
) -> str:
    request = result.request

    if (
        request is not None
        and request.action == "trash"
        and request.target_type == "file"
    ):
        return (
            "Okay, I won't move it to the Trash."
        )
    
    if (
        request is not None
        and request.action == "trash"
        and request.target_type == "files"
    ):
        return (
            "Okay, I won't move those files to the Trash."
        )
    return "Okay, I cancelled that action."

def _execution_data(
    result,
) -> dict:
    execution = getattr(
        result,
        "execution",
        None,
    )

    if execution is None:
        return {}

    if isinstance(
        execution,
        dict,
    ):
        return (
            execution.get("data")
            or {}
        )

    return (
        getattr(
            execution,
            "data",
            None,
        )
        or {}
    )