from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api.files import SaveFileRequest, notify_file_updated, validate_skill_content
from config import get_settings, runtime_config
from graph.agent import agent_manager
from memory_module_v2.service.config import get_memory_backend
from service.tool_security import get_tool_security_manager
from service.workbench_files import list_workbench_files, resolve_editable_path, write_text_file

router = APIRouter()


def _base_dir() -> Path:
    if agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return agent_manager.base_dir


@router.get("/workbench/files")
async def list_files() -> dict[str, Any]:
    return {"files": list_workbench_files(_base_dir())}


@router.post("/workbench/files")
async def create_file(payload: SaveFileRequest) -> dict[str, Any]:
    base = _base_dir()
    target = resolve_editable_path(base, payload.path)
    normalized = target.relative_to(base.resolve()).as_posix()
    if target.exists():
        raise HTTPException(status_code=409, detail="A file already exists at this path")
    validate_skill_content(normalized, payload.content)
    write_text_file(target, payload.content, create_only=True)
    notify_file_updated(normalized)
    return {"path": normalized, "ok": True}


@router.get("/workbench/runtime")
async def runtime_info() -> dict[str, Any]:
    base = _base_dir()
    settings = get_settings()
    policy = get_tool_security_manager(base).describe_policy()
    enabled = dict(policy["enabled_tools"])
    enabled["search_knowledge_base"] = runtime_config.get_rag_mode()
    for tool in agent_manager.tools:
        enabled.setdefault(tool.name, True)
    return {
        "python_version": platform.python_version(),
        "python_environment": "本地虚拟环境" if sys.prefix != sys.base_prefix else "系统 Python",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "memory_backend": get_memory_backend(),
        "rag_enabled": runtime_config.get_rag_mode(),
        "tools": [{"name": name, "enabled": value} for name, value in sorted(enabled.items())],
    }
