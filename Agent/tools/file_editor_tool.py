from pathlib import Path

WORKSPACE = Path(".").resolve()

# Paths (relative to repo root) that the file editor must never touch.
DENYLIST_RAW = [
    "__pycache__",
    "rag-env",
    "proxies.py",
    "demo_tool.py",
    ".git",
    ".gitignore",
]

# Resolve denylist to absolute paths for quick checks.
DENYLIST = []
for p in DENYLIST_RAW:
    try:
        DENYLIST.append((WORKSPACE / p).resolve())
    except Exception:
        # Fallback: construct Path and resolve
        DENYLIST.append(Path(p).resolve())

READ_FILE_TOOL = {

        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root."
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    }

WRITE_FILE_TOOL = {

        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file."
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": False
            }
        }
    }

EDIT_FILE_TOOL = {

        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description":("Path to the file, relative to the workspace root.")
                    },
                    "old_text": {
                        "type":("string"),
                        "description":("Text to replace.")
                    },
                   ("new_text"): {
                       ("type"):("string"),
                       ("description"):("New text to insert.")
                    }
                },
               ("required"):(["path", "old_text",("new_text")]),
               ("additionalProperties"): False
            }
        }
    }


FILE_EXISTS_TOOL = {
        "type": "function",
        "function": {
            "name": "file_exists",
            "description": (
                "Check whether a file or directory exists in the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root."
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    }

APPLY_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": (
            "Apply a unified diff patch to files in the workspace. "
            "The patch may modify multiple files. "
            "The operation is atomic: if any part of the patch fails, "
            "no files are modified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": (
                        "A unified diff containing the requested file changes. "
                        "Paths must be relative to the workspace root."
                    )
                }
            },
            "required": ["patch"],
            "additionalProperties": False
        }
    }
}


def safe_path(path: str) -> Path:
    # If an absolute path is provided, resolve it as-is. Otherwise resolve
    # relative paths against the workspace root for convenience.
    p = Path(path)
    if p.is_absolute():
        target = p.resolve()
    else:
        target = (WORKSPACE / path).resolve()

    # Deny access to specific files or directories (denylist uses absolute paths)
    for deny in DENYLIST:
        if target == deny or deny in target.parents:
            raise ValueError(f"Access to path is denied: {str(deny)}")

    return target


def file_exists(path: str):
    target = safe_path(path)

    return {
        "path": path,
        "exists": target.exists(),
        "type": (
            "directory"
            if target.is_dir()
            else "file"
            if target.is_file()
            else None
        ),
    }


def read_file(path: str):
    target = safe_path(path)

    if not target.is_file():
        raise FileNotFoundError(path)

    return {
        "path": path,
        "content": target.read_text(encoding="utf-8"),
    }


def write_file(path: str, content: str):
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return {
        "path": path,
        "success": True,
    }


def edit_file(path: str, old_text: str, new_text: str):
    target = safe_path(path)

    if not target.is_file():
        raise FileNotFoundError(path)

    content = target.read_text(encoding="utf-8")

    count = content.count(old_text)

    if count == 0:
        raise ValueError("old_text was not found")

    if count > 1:
        raise ValueError(
            f"old_text occurs {count} times; refusing ambiguous edit"
        )

    target.write_text(
        content.replace(old_text, new_text, 1),
        encoding="utf-8",
    )

    return {
        "path": path,
        "success": True,
    }

def parse_patch(patch: str):
    lines = patch.splitlines(keepends=True)

    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("Patch must start with *** Begin Patch")

    if lines[-1].strip() != "*** End Patch":
        raise ValueError("Patch must end with *** End Patch")

    operations = []
    i = 1

    while i < len(lines) - 1:
        line = lines[i]

        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: "):].strip()
            i += 1

            content = []

            while i < len(lines) - 1:
                if lines[i].startswith("*** "):
                    break

                if not lines[i].startswith("+"):
                    raise ValueError(
                        f"Added file line must start with '+': {lines[i]!r}"
                    )

                content.append(lines[i][1:])
                i += 1

            operations.append({
                "kind": "add",
                "path": path,
                "content": "".join(content),
            })

        elif line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: "):].strip()
            i += 1

            operations.append({
                "kind": "delete",
                "path": path,
            })

        elif line.startswith("*** Update File: "):
            path = line[len("*** Update File: "):].strip()
            i += 1

            hunks = []

            while i < len(lines) - 1:
                if lines[i].startswith("*** "):
                    break

                if not lines[i].startswith("@@"):
                    raise ValueError(
                        f"Expected hunk header, got: {lines[i]!r}"
                    )

                header = lines[i].rstrip("\n")
                i += 1

                hunk_lines = []

                while i < len(lines) - 1:
                    if lines[i].startswith("@@"):
                        break

                    if lines[i].startswith("*** "):
                        break

                    hunk_lines.append(lines[i])
                    i += 1

                hunks.append({
                    "header": header,
                    "lines": hunk_lines,
                })

            operations.append({
                "kind": "update",
                "path": path,
                "hunks": hunks,
            })

        elif line.strip() == "":
            i += 1

        else:
            raise ValueError(f"Unexpected patch line: {line!r}")

    return operations


def parse_patch(patch: str):
    lines = patch.splitlines(keepends=True)

    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("Patch must start with *** Begin Patch")

    if lines[-1].strip() != "*** End Patch":
        raise ValueError("Patch must end with *** End Patch")

    operations = []
    i = 1

    while i < len(lines) - 1:
        line = lines[i]

        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: "):].strip()
            i += 1

            content = []

            while i < len(lines) - 1:
                if lines[i].startswith("*** "):
                    break

                if not lines[i].startswith("+"):
                    raise ValueError(
                        f"Added file line must start with '+': {lines[i]!r}"
                    )

                content.append(lines[i][1:])
                i += 1

            operations.append({
                "kind": "add",
                "path": path,
                "content": "".join(content),
            })

        elif line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: "):].strip()
            i += 1

            operations.append({
                "kind": "delete",
                "path": path,
            })

        elif line.startswith("*** Update File: "):
            path = line[len("*** Update File: "):].strip()
            i += 1

            hunks = []

            while i < len(lines) - 1:
                if lines[i].startswith("*** "):
                    break

                if not lines[i].startswith("@@"):
                    raise ValueError(
                        f"Expected hunk header, got: {lines[i]!r}"
                    )

                header = lines[i].rstrip("\n")
                i += 1

                hunk_lines = []

                while i < len(lines) - 1:
                    if lines[i].startswith("@@"):
                        break

                    if lines[i].startswith("*** "):
                        break

                    hunk_lines.append(lines[i])
                    i += 1

                hunks.append({
                    "header": header,
                    "lines": hunk_lines,
                })

            operations.append({
                "kind": "update",
                "path": path,
                "hunks": hunks,
            })

        elif line.strip() == "":
            i += 1

        else:
            raise ValueError(f"Unexpected patch line: {line!r}")

    return operations


def apply_hunks(content: str, hunks: list[dict]) -> str:
    lines = content.splitlines(keepends=True)
    offset = 0

    for hunk in hunks:
        header = hunk["header"]

        match = re.match(
            r"@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?",
            header,
        )

        if not match:
            raise ValueError(f"Invalid hunk header: {header}")

        old_start = int(match.group(1))
        old_count = int(match.group(2) or 1)

        index = old_start - 1 + offset

        if index < 0 or index > len(lines):
            raise ValueError(
                f"Hunk starts outside file: {header}"
            )

        old_lines = []
        new_lines = []

        for line in hunk["lines"]:
            if not line:
                continue

            prefix = line[0]

            if prefix == " ":
                old_lines.append(line[1:])
                new_lines.append(line[1:])

            elif prefix == "-":
                old_lines.append(line[1:])

            elif prefix == "+":
                new_lines.append(line[1:])

            elif line.startswith("\\ No newline"):
                continue

            else:
                raise ValueError(
                    f"Invalid hunk line: {line!r}"
                )

        actual = lines[index:index + old_count]

        if actual != old_lines:
            raise ValueError(
                f"Hunk context does not match file at line {old_start}"
            )

        lines[index:index + old_count] = new_lines

        offset += len(new_lines) - old_count

    return "".join(lines)

def apply_patch(patch: str):
    """
    Apply a custom unified patch atomically.

    Supported operations:

        *** Add File: path
        *** Delete File: path
        *** Update File: path

    Update sections use @@ hunks with +/- lines.
    """

    operations = parse_patch(patch)

    # First construct every resulting file in memory.
    changes = []

    for operation in operations:
        path = safe_path(operation["path"])
        kind = operation["kind"]

        if kind == "add":
            if path.exists():
                raise ValueError(
                    f"Cannot add existing file: {operation['path']}"
                )

            new_content = operation["content"]
            changes.append(("write", path, new_content))

        elif kind == "delete":
            if not path.is_file():
                raise FileNotFoundError(operation["path"])

            changes.append(("delete", path, None))

        elif kind == "update":
            if not path.is_file():
                raise FileNotFoundError(operation["path"])

            old_content = path.read_text(encoding="utf-8")
            new_content = apply_hunks(
                old_content,
                operation["hunks"],
            )

            changes.append(("write", path, new_content))

    # Commit only after every operation has validated successfully.
    for kind, path, content in changes:
        if kind == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        elif kind == "delete":
            path.unlink()

    return {
        "success": True,
        "files_changed": [
            str(path.relative_to(WORKSPACE))
            for _, path, _ in changes
        ],
    }