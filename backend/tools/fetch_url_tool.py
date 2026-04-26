from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Type

import html2text
import httpx
from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from service.tool_security import ToolSecurityManager, get_tool_security_manager


class FetchURLInput(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch")


class FetchURLTool(BaseTool):
    name: str = "fetch_url"
    description: str = "Fetch a URL. JSON stays JSON; HTML is converted into markdown-like plain text."
    args_schema: Type[BaseModel] = FetchURLInput
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

    def _format_response(self, response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            return json.dumps(response.json(), ensure_ascii=False, indent=2)[:5000]
        if "html" in content_type:
            parser = html2text.HTML2Text()
            parser.ignore_links = False
            parser.ignore_images = True
            return parser.handle(response.text)[:5000]
        return response.text[:5000]

    def _run(
        self,
        url: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        started = time.perf_counter()
        decision = self._security.check_fetch_url(url)
        if not decision.allowed:
            message = f"Fetch failed: {decision.reason}"
            self._security.audit.record(
                tool=self.name,
                status="blocked",
                input_value=url,
                reason=decision.reason,
            )
            return message
        try:
            with httpx.Client(follow_redirects=True, timeout=15) as client:
                response = client.get(url)
                response.raise_for_status()
        except Exception as exc:
            message = f"Fetch failed: {exc}"
            self._security.audit.record(
                tool=self.name,
                status="error",
                input_value=url,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message
        output = self._format_response(response)
        self._security.audit.record(
            tool=self.name,
            status="success",
            input_value=url,
            output_value=output,
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={"status_code": response.status_code},
        )
        return output

    async def _arun(
        self,
        url: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        started = time.perf_counter()
        decision = self._security.check_fetch_url(url)
        if not decision.allowed:
            message = f"Fetch failed: {decision.reason}"
            self._security.audit.record(
                tool=self.name,
                status="blocked",
                input_value=url,
                reason=decision.reason,
            )
            return message
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                response = await client.get(url)
                response.raise_for_status()
        except Exception as exc:
            message = f"Fetch failed: {exc}"
            self._security.audit.record(
                tool=self.name,
                status="error",
                input_value=url,
                output_value=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return message
        output = self._format_response(response)
        self._security.audit.record(
            tool=self.name,
            status="success",
            input_value=url,
            output_value=output,
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={"status_code": response.status_code},
        )
        return output
