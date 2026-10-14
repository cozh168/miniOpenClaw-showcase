from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from graph.agent import agent_manager
from memory_module_v2.service.config import get_memory_backend
from service.memory_indexer import memory_indexer
from service.workbench_files import read_text_file, resolve_editable_path, validate_text_content, write_text_file
from tools.skills_scanner import refresh_snapshot, scan_skills, validate_skill_frontmatter

router = APIRouter()

class SaveFileRequest(BaseModel):
    path: str = Field(..., min_length=1)
    content: str


_DEFAULT_TEMPLATES: dict[str, str] = {
    "workspace/AGENTS.md": (
        "# Agents Guide\n\n"
        "Describe workspace-specific agent rules, tool usage conventions, and safety notes here.\n"
    ),
    "workspace/IDENTITY.md": (
        "# Workspace Identity\n\n"
        "Define the project identity, tone, and operating style here.\n"
    ),
    "workspace/SOUL.md": (
        "# Soul\n\n"
        "Describe the assistant's long-term goals and behavior principles here.\n"
    ),
    "workspace/USER.md": (
        "# User Profile\n\n"
        "Record user preferences only after the user explicitly provides them.\n"
    ),
    "memory_module_v1/long_term_memory/MEMORY.md": (
        "# Long-term Memory\n\n"
        "Add durable memory snippets here only after they are explicitly confirmed.\n"
    ),
}


def _resolve_path(relative_path: str) -> Path:
    if agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    return resolve_editable_path(agent_manager.base_dir, relative_path)


def notify_file_updated(normalized: str) -> None:
    if normalized == "memory_module_v1/long_term_memory/MEMORY.md" and get_memory_backend() == "v1":
        memory_indexer.rebuild_index()
    if normalized.startswith("skills/"):
        refresh_snapshot(agent_manager.base_dir)


def validate_skill_content(normalized: str, content: str) -> None:
    if normalized.startswith("skills/") and Path(normalized).name.casefold() == "skill.md":
        validate_text_content(content)
        try:
            validate_skill_frontmatter(content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/files")
async def read_file(path: str = Query(..., min_length=1)) -> dict[str, str]:
    file_path = _resolve_path(path)
    normalized = file_path.relative_to(agent_manager.base_dir.resolve()).as_posix()
    if not file_path.exists():
        default_content = _DEFAULT_TEMPLATES.get(normalized)
        if default_content is not None:
            write_text_file(file_path, default_content)
        else:
            raise HTTPException(status_code=404, detail="File not found")
    return {
        "path": normalized,
        "content": read_text_file(file_path),
    }


@router.post("/files")
async def save_file(payload: SaveFileRequest) -> dict[str, Any]:
    file_path = _resolve_path(payload.path)
    normalized = file_path.relative_to(agent_manager.base_dir.resolve()).as_posix()
    validate_skill_content(normalized, payload.content)
    write_text_file(file_path, payload.content)
    notify_file_updated(normalized)

    return {"ok": True, "path": normalized}


@router.get("/skills")
async def list_skills() -> list[dict[str, str]]:
    if agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return [skill.__dict__ for skill in scan_skills(agent_manager.base_dir)]
