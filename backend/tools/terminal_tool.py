from __future__ import annotations

import asyncio
import platform
import subprocess
import time
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from config import get_settings
from service.tool_security import ToolSecurityManager, get_tool_security_manager


class TerminalToolInput(BaseModel):
    command: str = Field(..., description="Shell command to execute inside the project root")


class TerminalTool(BaseTool):
    name: str = "terminal"
    description: str = (
        "Execute shell commands inside the project root. Use this for quick inspection, "
        "building, or local commands. Dangerous system-destructive commands are blocked."
    )
    args_schema: Type[BaseModel] = TerminalToolInput
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

    def _run(
        self,
        command: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        started = time.perf_counter()
        decision = self._security.check_terminal_command(command)
        if not decision.allowed:
            message = f"Blocked: {decision.reason}"
            self._security.audit.record(
                tool=self.name,
                status="blocked",
                input_value=command,
                reason=decision.reason,
            )
            return message

        settings = get_settings()
        shell_command = (
            ["powershell", "-NoProfile", "-Command", command]
            if platform.system().lower().startswith("win")
            else ["bash", "-lc", command]
        )
        try:
            completed = subprocess.run(
                shell_command,
                cwd=self._root_dir,
                capture_output=True,
                text=True,
                timeout=settings.terminal_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            message = "Timed out after 30 seconds."
            self._security.audit.record(
                tool=self.name,
                status="timeout",
                input_value=command,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message

        combined = (completed.stdout or "") + (completed.stderr or "")
        combined = combined.strip() or "[no output]"
        output = combined[:5000]
        self._security.audit.record(
            tool=self.name,
            status="success" if completed.returncode == 0 else "error",
            input_value=command,
            output_value=output,
            reason=None if completed.returncode == 0 else f"exit code {completed.returncode}",
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={"returncode": completed.returncode},
        )
        return output

    async def _arun(
        self,
        command: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, command, None)
