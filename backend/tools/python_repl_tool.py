from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from service.tool_security import ToolSecurityManager, get_tool_security_manager


class PythonReplInput(BaseModel):
    code: str = Field(..., description="Python code to execute")


class PythonReplTool(BaseTool):
    name: str = "python_repl"
    description: str = "Execute short Python snippets in a subprocess and return stdout/stderr."
    args_schema: Type[BaseModel] = PythonReplInput
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
        code: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        started = time.perf_counter()
        decision = self._security.check_python_repl()
        if not decision.allowed:
            message = f"Blocked: {decision.reason}"
            self._security.audit.record(
                tool=self.name,
                status="blocked",
                input_value=code,
                reason=decision.reason,
            )
            return message
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self._root_dir,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            message = "Timed out after 15 seconds."
            self._security.audit.record(
                tool=self.name,
                status="timeout",
                input_value=code,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message
        combined = (completed.stdout or "") + (completed.stderr or "")
        output = (combined.strip() or "[no output]")[:5000]
        self._security.audit.record(
            tool=self.name,
            status="success" if completed.returncode == 0 else "error",
            input_value=code,
            output_value=output,
            reason=None if completed.returncode == 0 else f"exit code {completed.returncode}",
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={"returncode": completed.returncode},
        )
        return output

    async def _arun(
        self,
        code: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, code, None)
