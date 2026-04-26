from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from service.tool_security import ToolSecurityManager, get_tool_security_manager


class ReadFileInput(BaseModel):
    path: str = Field(..., description="Relative path inside the project root")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read a local file under the project root. Use relative paths like skills/foo/SKILL.md."
    args_schema: Type[BaseModel] = ReadFileInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()
    _security: ToolSecurityManager = PrivateAttr()

    def __init__(
        self,
        root_dir: Path,
        *,
        security_manager: ToolSecurityManager | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir.resolve()
        self._security = security_manager or get_tool_security_manager(self._root_dir)

    def _resolve_path(self, path: str) -> Path:
        candidate = (self._root_dir / path).resolve()
        if self._root_dir not in candidate.parents and candidate != self._root_dir:
            raise ValueError("Path traversal detected.")
        return candidate

    def _run(
        self,
        path: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        started = time.perf_counter()
        try:
            file_path = self._resolve_path(path)
        except ValueError as exc:
            message = f"Read failed: {exc}"
            self._security.audit.record(
                tool=self.name,
                status="blocked",
                input_value=path,
                reason=str(exc),
            )
            return message
        decision = self._security.check_read_path(file_path)
        if not decision.allowed:
            message = f"Read failed: {decision.reason}"
            self._security.audit.record(
                tool=self.name,
                status="blocked",
                input_value=path,
                reason=decision.reason,
            )
            return message
        if not file_path.exists():
            message = "Read failed: file does not exist."
            self._security.audit.record(
                tool=self.name,
                status="error",
                input_value=path,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message
        if file_path.is_dir():
            message = "Read failed: path is a directory."
            self._security.audit.record(
                tool=self.name,
                status="error",
                input_value=path,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message
        try:
            output = file_path.read_text(encoding="utf-8")[:10000]
        except UnicodeDecodeError:
            message = "Read failed: file is not valid UTF-8 text."
            self._security.audit.record(
                tool=self.name,
                status="error",
                input_value=path,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message
        self._security.audit.record(
            tool=self.name,
            status="success",
            input_value=path,
            output_value=output,
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={"resolved_path": str(file_path)},
        )
        return output

    async def _arun(
        self,
        path: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, path, None)
