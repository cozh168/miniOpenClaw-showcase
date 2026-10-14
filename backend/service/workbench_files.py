from __future__ import annotations

import heapq
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException

MAX_TEXT_BYTES = 1024 * 1024
MAX_TREE_NODES = 500
WORKBENCH_ROOTS = (
    "workspace",
    "knowledge",
    "skills",
    "memory_module_v1/long_term_memory",
    "memory",
)
EDITABLE_ROOTS = WORKBENCH_ROOTS
TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".xml", ".toml",
    ".ini", ".cfg", ".sql", ".sh", ".r", ".rst", ".log",
}
_ALIASES = {
    "SKILLS_SNAPSHOT.md": "skills/SKILLS_SNAPSHOT.md",
    "MEMORY.md": "memory_module_v1/long_term_memory/MEMORY.md",
}


def _safe_name(name: str) -> bool:
    try:
        if len(name.encode("utf-8")) > 255:
            return False
    except UnicodeEncodeError:
        return False
    lowered = name.casefold()
    sensitive = ("secret", "credential", "password", "api_key", "apikey", "api-key", "private_key", "private-key")
    return (
        bool(name)
        and name not in {".", ".."}
        and not name.startswith(".")
        and not name.endswith((" ", "."))
        and not any(ord(char) < 32 or ord(char) == 127 or char in '<>:"|?*' for char in name)
        and not any(word in lowered for word in sensitive)
        and lowered not in {"id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"}
        and not lowered.endswith((".pem", ".key", ".p12", ".pfx", ".env"))
    )


def resolve_editable_path(base_dir: Path, relative_path: str) -> Path:
    normalized = _ALIASES.get(relative_path, relative_path).replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) > 33 or not all(_safe_name(part) for part in parts):
        raise HTTPException(status_code=400, detail="Invalid or sensitive file path")
    if not any(normalized.startswith(root + "/") for root in EDITABLE_ROOTS):
        raise HTTPException(status_code=400, detail="Path is not in the editable whitelist")

    base = base_dir.resolve()
    candidate = base.joinpath(*parts)
    current = base
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise HTTPException(status_code=400, detail="Symbolic links are not supported")
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid file path") from exc
    if not resolved.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Path traversal detected")
    return resolved


def validate_text_content(content: str) -> bytes:
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise HTTPException(status_code=400, detail="File must contain valid UTF-8 text") from exc
    if len(encoded) > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="Text file exceeds the 1 MB limit")
    if "\x00" in content:
        raise HTTPException(status_code=400, detail="Binary files cannot be edited as text")
    return encoded


def validate_text_file(path: Path, *, must_exist: bool = True) -> None:
    if path.is_dir():
        raise HTTPException(status_code=400, detail="Select a file instead of a directory")
    if path.suffix.casefold() not in TEXT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="This file type cannot be edited as text")
    if not path.exists():
        if must_exist:
            raise HTTPException(status_code=404, detail="File not found")
        return
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Only regular files can be edited")
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="Text file exceeds the 1 MB limit")


def read_text_file(path: Path) -> str:
    validate_text_file(path)
    try:
        # Read a bounded number of bytes even if the file changes after stat().
        with path.open("rb") as handle:
            data = handle.read(MAX_TEXT_BYTES + 1)
        if len(data) > MAX_TEXT_BYTES:
            raise HTTPException(status_code=413, detail="Text file exceeds the 1 MB limit")
        content = data.decode("utf-8")
        validate_text_content(content)
        return content
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File is binary or is not UTF-8 text") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="File could not be read") from exc


def write_text_file(path: Path, content: str, *, create_only: bool = False) -> None:
    encoded = validate_text_content(content)
    if create_only and path.exists():
        raise HTTPException(status_code=409, detail="A file already exists at this path")
    validate_text_file(path, must_exist=False)
    if path.exists():
        read_text_file(path)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a fully written file. A failed write must not truncate the
        # original, and two simultaneous create requests cannot overwrite each other.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".workbench-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        if create_only:
            os.link(temporary, path, follow_symlinks=False)
        else:
            os.replace(temporary, path)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="A file already exists at this path") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="File could not be saved at this path") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def list_workbench_files(base_dir: Path, *, limit: int = MAX_TREE_NODES) -> list[dict[str, Any]]:
    base = base_dir.resolve()
    count = 0

    def visit(path: Path, depth: int = 0, *, root: bool = False) -> dict[str, Any] | None:
        nonlocal count
        if (count >= limit and not root) or depth > 32 or path.is_symlink() or not _safe_name(path.name):
            return None
        try:
            if not path.resolve().is_relative_to(base):
                return None
            if not path.is_dir() and not path.is_file():
                return None
            is_directory = path.is_dir()
            size = 0 if is_directory else path.stat().st_size
            if not root:
                count += 1
            node: dict[str, Any] = {
                "path": path.relative_to(base).as_posix(),
                "name": path.name,
                "type": "directory" if is_directory else "file",
                "size": size,
                "editable": False,
            }
            if is_directory:
                # Keep at most the remaining node budget in memory, even when
                # a directory contains thousands of entries. DirEntry caches
                # type metadata and skips links and hidden names before sorting.
                def eligible_entries(iterator):
                    for entry in iterator:
                        if not _safe_name(entry.name) or entry.is_symlink():
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False) or entry.is_file(follow_symlinks=False):
                                yield entry
                        except OSError:
                            continue

                with os.scandir(path) as iterator:
                    entries = heapq.nsmallest(
                        max(0, limit - count),
                        eligible_entries(iterator),
                        key=lambda entry: (not entry.is_dir(follow_symlinks=False), entry.name.casefold(), entry.name),
                    )
                node["children"] = []
                for entry in entries:
                    child = visit(Path(entry.path), depth + 1)
                    if child is not None:
                        node["children"].append(child)
                    if count >= limit:
                        break
            elif path.suffix.casefold() in TEXT_EXTENSIONS and size <= MAX_TEXT_BYTES:
                try:
                    read_text_file(path)
                    node["editable"] = True
                except HTTPException:
                    pass
            return node
        except OSError:
            return None

    roots = []
    for path in sorted((base / root for root in WORKBENCH_ROOTS), key=lambda path: path.name.casefold()):
        # Check intermediate components too (long_term_memory has a parent root).
        if not path.exists() or any(parent.is_symlink() for parent in [path, *path.parents] if parent != base and parent.is_relative_to(base)):
            continue
        roots.append(path)
    # Reserve the real top-level roots so a large knowledge folder cannot hide
    # Workspace, Skills and Memory entirely when the total limit is reached.
    roots = roots[:max(0, limit)]
    count = len(roots)
    result = []
    for path in roots:
        node = visit(path, root=True)
        if node is not None:
            result.append(node)
    return result
