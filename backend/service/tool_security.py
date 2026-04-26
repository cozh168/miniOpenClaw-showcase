from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from config import Settings, get_settings

DEFAULT_TERMINAL_BLOCKED_PATTERNS = (
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "format ",
    ":(){:|:&};:",
)
DEFAULT_READ_FILE_BLOCKED_PREFIXES = (
    ".git/",
    ".venv/",
    "backend/config/.env",
    "backend/storage/",
    "backend/sessions/",
    "frontend/node_modules/",
    "__pycache__/",
)
DEFAULT_FETCH_URL_BLOCKED_HOSTS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
)
DEFAULT_FETCH_URL_BLOCKED_SUFFIXES = (
    ".local",
    ".internal",
)
MAX_PREVIEW_CHARS = 300


def _normalize_csv(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip().lower() for value in values if value.strip())


def _normalize_path_rule(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def _path_matches(path: str, rule: str) -> bool:
    normalized_path = _normalize_path_rule(path)
    normalized_rule = _normalize_path_rule(rule)
    return normalized_path == normalized_rule or normalized_path.startswith(f"{normalized_rule}/")


def _host_matches(host: str, rule: str) -> bool:
    normalized_host = host.strip().lower()
    normalized_rule = rule.strip().lower().lstrip(".")
    return normalized_host == normalized_rule or normalized_host.endswith(f".{normalized_rule}")


def _truncate(value: Any, limit: int = MAX_PREVIEW_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _is_private_or_local_host(host: str) -> bool:
    try:
        candidate = ip_address(host)
    except ValueError:
        return False
    return (
        candidate.is_private
        or candidate.is_loopback
        or candidate.is_link_local
        or candidate.is_multicast
        or candidate.is_reserved
        or candidate.is_unspecified
    )


@dataclass(frozen=True)
class ToolAccessDecision:
    allowed: bool
    reason: str | None = None


class ToolAuditLog:
    def __init__(self, root_dir: Path, *, enabled: bool, max_entries: int) -> None:
        self._enabled = enabled
        self._max_entries = max(1, max_entries)
        self._log_path = root_dir.resolve() / "storage" / "tool_audit" / "tool_runs.jsonl"
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._log_path

    def record(
        self,
        *,
        tool: str,
        status: str,
        input_value: Any = None,
        output_value: Any = None,
        reason: str | None = None,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "status": status,
            "input_preview": _truncate(input_value),
            "output_preview": _truncate(output_value),
            "reason": reason,
            "duration_ms": duration_ms,
            "metadata": metadata or {},
        }

        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._trim_locked()

    def list_entries(
        self,
        *,
        limit: int = 100,
        tool: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._log_path.exists():
            return []

        normalized_tool = tool.strip().lower() if tool else None
        normalized_status = status.strip().lower() if status else None
        entries: list[dict[str, Any]] = []

        with self._lock:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if normalized_tool and str(entry.get("tool", "")).lower() != normalized_tool:
                continue
            if normalized_status and str(entry.get("status", "")).lower() != normalized_status:
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        return entries

    def _trim_locked(self) -> None:
        if not self._log_path.exists():
            return
        lines = self._log_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self._max_entries * 2:
            return
        trimmed = "\n".join(lines[-self._max_entries :])
        if trimmed:
            trimmed += "\n"
        self._log_path.write_text(trimmed, encoding="utf-8")


class ToolSecurityManager:
    def __init__(self, root_dir: Path, settings: Settings | None = None) -> None:
        self.root_dir = root_dir.resolve()
        self.settings = settings or get_settings()
        self.audit = ToolAuditLog(
            self.root_dir,
            enabled=self.settings.tool_audit_enabled,
            max_entries=self.settings.tool_audit_max_entries,
        )
        self._terminal_allowed = _normalize_csv(self.settings.terminal_allowed_commands)
        self._read_allowed = tuple(
            _normalize_path_rule(item) for item in self.settings.read_file_allowed_prefixes if item.strip()
        )
        self._read_blocked = tuple(
            _normalize_path_rule(item)
            for item in (
                *DEFAULT_READ_FILE_BLOCKED_PREFIXES,
                *self.settings.read_file_blocked_prefixes,
            )
            if item.strip()
        )
        self._fetch_allowed_hosts = _normalize_csv(self.settings.fetch_url_allowed_hosts)
        self._fetch_blocked_hosts = _normalize_csv(
            (*DEFAULT_FETCH_URL_BLOCKED_HOSTS, *self.settings.fetch_url_blocked_hosts)
        )

    def is_tool_enabled(self, tool_name: str) -> bool:
        normalized = tool_name.strip().lower()
        flags = {
            "terminal": self.settings.terminal_tool_enabled,
            "python_repl": self.settings.python_repl_tool_enabled,
            "fetch_url": self.settings.fetch_url_tool_enabled,
            "read_file": self.settings.read_file_tool_enabled,
        }
        return flags.get(normalized, True)

    def check_tool_enabled(self, tool_name: str) -> ToolAccessDecision:
        if self.is_tool_enabled(tool_name):
            return ToolAccessDecision(True)
        return ToolAccessDecision(False, f"{tool_name} tool is disabled by configuration.")

    def check_terminal_command(self, command: str) -> ToolAccessDecision:
        enabled = self.check_tool_enabled("terminal")
        if not enabled.allowed:
            return enabled

        lowered = command.lower()
        if any(pattern in lowered for pattern in DEFAULT_TERMINAL_BLOCKED_PATTERNS):
            return ToolAccessDecision(False, "command matches the terminal blacklist.")

        if not self._terminal_allowed:
            return ToolAccessDecision(True)

        command_name = command.strip().split(maxsplit=1)[0].lower()
        if command_name in self._terminal_allowed:
            return ToolAccessDecision(True)
        return ToolAccessDecision(
            False,
            f"command '{command_name}' is not in TERMINAL_ALLOWED_COMMANDS.",
        )

    def check_python_repl(self) -> ToolAccessDecision:
        return self.check_tool_enabled("python_repl")

    def check_read_path(self, target_path: Path) -> ToolAccessDecision:
        enabled = self.check_tool_enabled("read_file")
        if not enabled.allowed:
            return enabled

        relative_path = target_path.resolve().relative_to(self.root_dir).as_posix()
        if any(_path_matches(relative_path, rule) for rule in self._read_blocked):
            return ToolAccessDecision(False, "requested path is blocked by the read-file policy.")

        if self._read_allowed and not any(
            _path_matches(relative_path, rule) for rule in self._read_allowed
        ):
            return ToolAccessDecision(False, "requested path is outside READ_FILE_ALLOWED_PREFIXES.")

        return ToolAccessDecision(True)

    def check_fetch_url(self, url: str) -> ToolAccessDecision:
        enabled = self.check_tool_enabled("fetch_url")
        if not enabled.allowed:
            return enabled

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ToolAccessDecision(False, "only http and https URLs are allowed.")

        host = (parsed.hostname or "").strip().lower()
        if not host:
            return ToolAccessDecision(False, "URL is missing a hostname.")

        if any(_host_matches(host, blocked_host) for blocked_host in self._fetch_blocked_hosts):
            return ToolAccessDecision(False, "target host is blocked by the fetch policy.")

        if any(host.endswith(suffix) for suffix in DEFAULT_FETCH_URL_BLOCKED_SUFFIXES):
            return ToolAccessDecision(False, "target host suffix is blocked by the fetch policy.")

        if self._fetch_allowed_hosts and not any(
            _host_matches(host, allowed_host) for allowed_host in self._fetch_allowed_hosts
        ):
            return ToolAccessDecision(False, "target host is outside FETCH_URL_ALLOWED_HOSTS.")

        if not self.settings.fetch_url_allow_private_hosts and _is_private_or_local_host(host):
            return ToolAccessDecision(False, "private or local IP targets are not allowed.")

        return ToolAccessDecision(True)

    def describe_policy(self) -> dict[str, Any]:
        return {
            "enabled_tools": {
                "terminal": self.settings.terminal_tool_enabled,
                "python_repl": self.settings.python_repl_tool_enabled,
                "fetch_url": self.settings.fetch_url_tool_enabled,
                "read_file": self.settings.read_file_tool_enabled,
            },
            "terminal_allowed_commands": list(self._terminal_allowed),
            "terminal_blocked_patterns": list(DEFAULT_TERMINAL_BLOCKED_PATTERNS),
            "read_file_allowed_prefixes": list(self._read_allowed),
            "read_file_blocked_prefixes": list(self._read_blocked),
            "fetch_url_allowed_hosts": list(self._fetch_allowed_hosts),
            "fetch_url_blocked_hosts": list(self._fetch_blocked_hosts),
            "fetch_url_allow_private_hosts": self.settings.fetch_url_allow_private_hosts,
            "tool_audit_enabled": self.settings.tool_audit_enabled,
            "tool_audit_max_entries": self.settings.tool_audit_max_entries,
            "tool_audit_log_path": str(self.audit.path),
        }


@lru_cache(maxsize=8)
def get_tool_security_manager(root_dir: Path) -> ToolSecurityManager:
    return ToolSecurityManager(root_dir)
