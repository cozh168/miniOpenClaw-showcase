from __future__ import annotations

import math
import re
from typing import Any

import tiktoken
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from graph.agent import agent_manager
from service.prompt_builder import build_system_prompt

router = APIRouter()

_ENCODER = None
_ENCODER_INIT_FAILED = False
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class FileTokensRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


def _get_encoder():
    global _ENCODER_INIT_FAILED, _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    if _ENCODER_INIT_FAILED:
        return None

    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fall back to a local heuristic when tiktoken cannot download/cache
        # its encoding file in restricted environments.
        _ENCODER_INIT_FAILED = True
        return None
    return _ENCODER


def _estimate_tokens(text: str) -> int:
    content = text or ""
    if not content:
        return 0

    cjk_count = len(_CJK_RE.findall(content))
    other_count = len(content) - cjk_count
    return max(1, cjk_count + math.ceil(other_count / 4))


def _count_tokens(text: str) -> int:
    encoder = _get_encoder()
    if encoder is None:
        return _estimate_tokens(text)
    return len(encoder.encode(text or ""))


@router.get("/tokens/session/{session_id}")
async def session_tokens(session_id: str) -> dict[str, int]:
    session_manager = agent_manager.session_manager
    if session_manager is None or agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    record = session_manager.get_history(session_id)
    system_prompt = build_system_prompt(agent_manager.base_dir)
    message_text = []
    for item in record.get("messages", []):
        message_text.append(str(item.get("content", "")))
        for tool_call in item.get("tool_calls", []) or []:
            message_text.append(str(tool_call))

    system_tokens = _count_tokens(system_prompt)
    message_tokens = _count_tokens("\n".join(message_text))
    return {
        "system_tokens": system_tokens,
        "message_tokens": message_tokens,
        "total_tokens": system_tokens + message_tokens,
    }


@router.post("/tokens/files")
async def file_tokens(payload: FileTokensRequest) -> dict[str, Any]:
    if agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    files: list[dict[str, Any]] = []
    total = 0
    for relative_path in payload.paths:
        path = (agent_manager.base_dir / relative_path).resolve()
        if not path.exists() or path.is_dir():
            continue
        count = _count_tokens(path.read_text(encoding="utf-8"))
        total += count
        files.append({"path": relative_path, "tokens": count})

    return {"files": files, "total_tokens": total}
