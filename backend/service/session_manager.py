from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

DEFAULT_SESSION_TITLE = "New Chat"


def _publish_json(path: Path, record: dict[str, Any], *, create_only: bool = False) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".session-", mode="w", encoding="utf-8", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if create_only:
            os.link(temporary, path, follow_symlinks=False)
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SessionBusyError(RuntimeError):
    pass


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class SessionManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._lock = threading.RLock()
        self._active_chats: dict[str, str] = {}
        self.sessions_dir = base_dir / "memory_module_v1" / "sessions"
        self.archive_dir = self.sessions_dir / "archive"
        self.trash_dir = self.sessions_dir / "trash"
        self._check_storage_paths()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def _check_storage_paths(self) -> None:
        for path in (self.base_dir / "memory_module_v1", self.sessions_dir, self.archive_dir, self.trash_dir):
            if path.is_symlink():
                raise ValueError("Symbolic session directories are not supported")

    def _session_path(self, session_id: str) -> Path:
        self._check_storage_paths()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("Invalid session ID")
        path = self.sessions_dir / f"{session_id}.json"
        if path.is_symlink():
            raise ValueError("Symbolic session paths are not supported")
        return path

    def _trash_path(self, session_id: str) -> Path:
        path = self.trash_dir / self._session_path(session_id).name
        if path.is_symlink():
            raise ValueError("Symbolic session paths are not supported")
        return path

    @_synchronized
    def session_exists(self, session_id: str) -> bool:
        return self._session_path(session_id).is_file()

    @_synchronized
    def begin_chat(self, session_id: str) -> str:
        self.load_session_record(session_id)
        if session_id in self._active_chats:
            raise SessionBusyError("This session is already generating a response")
        token = uuid.uuid4().hex
        self._active_chats[session_id] = token
        return token

    @_synchronized
    def end_chat(self, session_id: str, token: str) -> None:
        if self._active_chats.get(session_id) == token:
            del self._active_chats[session_id]

    def _default_record(
        self,
        session_id: str,
        title: str = DEFAULT_SESSION_TITLE,
    ) -> dict[str, Any]:
        now = time.time()
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "compressed_context": "",
            "messages": [],
        }

    @_synchronized
    def _read_session_file(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError("Session not found")

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            record = self._default_record(session_id)
            record["messages"] = raw
            self._write_session(record)
            return record

        if not isinstance(raw, dict) or not isinstance(raw.get("messages", []), list):
            raise ValueError("Invalid session record")
        # The filename owns the identity, including migrated legacy records.
        raw["id"] = session_id
        raw.setdefault("title", DEFAULT_SESSION_TITLE)
        raw.setdefault("created_at", time.time())
        raw.setdefault("updated_at", raw["created_at"])
        raw.setdefault("compressed_context", "")
        raw.setdefault("messages", [])
        return raw

    @_synchronized
    def _write_session(self, record: dict[str, Any]) -> None:
        session_id = str(record["id"])
        record["updated_at"] = time.time()
        _publish_json(self._session_path(session_id), record)

    @_synchronized
    def create_session(self, title: str = DEFAULT_SESSION_TITLE) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        record = self._default_record(session_id, title=title)
        self._write_session(record)
        return record

    @_synchronized
    def _list_records(self, directory: Path) -> list[dict[str, Any]]:
        self._check_storage_paths()
        records: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            if path.is_symlink():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeError, OSError):
                continue
            if isinstance(record, list):
                record = {"id": path.stem, "messages": record}
            if not isinstance(record, dict) or not isinstance(record.get("messages", []), list):
                continue
            records.append(
                {
                    "id": path.stem,
                    "title": record.get("title", DEFAULT_SESSION_TITLE),
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "message_count": len(record.get("messages", [])),
                    **({"deleted_at": record.get("deleted_at")} if directory == self.trash_dir else {}),
                }
            )
        return sorted(records, key=lambda item: item.get("updated_at") or 0, reverse=True)

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._list_records(self.sessions_dir)

    def list_trash(self) -> list[dict[str, Any]]:
        return self._list_records(self.trash_dir)

    @_synchronized
    def load_session_record(self, session_id: str) -> dict[str, Any]:
        if not self.session_exists(session_id):
            raise FileNotFoundError("Session not found")
        return self._read_session_file(session_id)

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        return self._read_session_file(session_id)["messages"]

    def load_session_for_agent(self, session_id: str) -> list[dict[str, str]]:
        record = self.load_session_record(session_id)
        merged: list[dict[str, str]] = []

        compressed_context = record.get("compressed_context", "").strip()
        if compressed_context:
            merged.append(
                {
                    "role": "assistant",
                    "content": f"[Previous conversation summary]\n{compressed_context}",
                }
            )

        for message in record.get("messages", []):
            role = message.get("role", "")
            content = str(message.get("content", "") or "")
            if role == "assistant" and merged and merged[-1]["role"] == "assistant":
                if content:
                    if merged[-1]["content"]:
                        merged[-1]["content"] += "\n\n" + content
                    else:
                        merged[-1]["content"] = content
                continue

            merged.append({"role": role, "content": content})

        return [item for item in merged if item["role"] in {"user", "assistant"}]

    @_synchronized
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        record = self._read_session_file(session_id)
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        record["messages"].append(message)
        self._write_session(record)
        return message

    def get_history(self, session_id: str) -> dict[str, Any]:
        return self._read_session_file(session_id)

    @_synchronized
    def save_exchange(self, session_id: str, user_content: str, segments: list[dict[str, Any]]) -> None:
        record = self._read_session_file(session_id)
        record["messages"].append({"role": "user", "content": user_content})
        for segment in segments:
            message: dict[str, Any] = {"role": "assistant", "content": segment["content"]}
            if segment.get("tool_calls"):
                message["tool_calls"] = segment["tool_calls"]
            record["messages"].append(message)
        self._write_session(record)

    @_synchronized
    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        record = self._read_session_file(session_id)
        record["title"] = title.strip() or DEFAULT_SESSION_TITLE
        self._write_session(record)
        return record

    def set_title(self, session_id: str, title: str) -> dict[str, Any]:
        return self.rename_session(session_id, title)

    @_synchronized
    def delete_session(self, session_id: str) -> None:
        if session_id in self._active_chats:
            raise SessionBusyError("Wait for the response to finish before deleting this session")
        path = self._session_path(session_id)
        if not path.is_file():
            raise FileNotFoundError("Session not found")
        target = self._trash_path(session_id)
        if target.exists():
            raise FileExistsError("A deleted session with this ID already exists")
        record = self._read_session_file(session_id)
        record["deleted_at"] = time.time()
        # Exclusive creation prevents an existing deleted session from being overwritten.
        _publish_json(target, record, create_only=True)
        path.unlink()

    @_synchronized
    def restore_session(self, session_id: str) -> dict[str, Any]:
        if session_id in self._active_chats:
            raise SessionBusyError("Wait for the response to finish before restoring this session")
        target = self._session_path(session_id)
        source = self._trash_path(session_id)
        if not source.is_file():
            raise FileNotFoundError("Deleted session not found")
        if target.exists():
            raise FileExistsError("A session with this ID already exists")
        record = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not isinstance(record.get("messages", []), list):
            raise ValueError("Invalid deleted session record")
        record["id"] = session_id
        record.pop("deleted_at", None)
        record["updated_at"] = time.time()
        _publish_json(target, record, create_only=True)
        source.unlink()
        return record

    @_synchronized
    def compress_history(self, session_id: str, summary: str, n_messages: int) -> dict[str, int]:
        record = self._read_session_file(session_id)
        messages = record.get("messages", [])
        archived = messages[:n_messages]
        remaining = messages[n_messages:]

        archive_path = self.archive_dir / f"{session_id}_{time.time_ns()}_{uuid.uuid4().hex[:8]}.json"
        archive_payload = {
            "session_id": session_id,
            "archived_at": time.time(),
            "messages": archived,
        }
        _publish_json(archive_path, archive_payload, create_only=True)

        existing_summary = record.get("compressed_context", "").strip()
        if existing_summary:
            record["compressed_context"] = f"{existing_summary}\n---\n{summary.strip()}"
        else:
            record["compressed_context"] = summary.strip()
        record["messages"] = remaining
        self._write_session(record)
        return {
            "archived_count": len(archived),
            "remaining_count": len(remaining),
        }

    def get_compressed_context(self, session_id: str) -> str:
        return self._read_session_file(session_id).get("compressed_context", "")
