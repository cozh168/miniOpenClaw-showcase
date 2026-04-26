from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from graph.agent import agent_manager
from service.tool_security import get_tool_security_manager

router = APIRouter()


def _security_manager():
    if agent_manager.base_dir is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return get_tool_security_manager(agent_manager.base_dir)


@router.get("/tool-security")
async def get_tool_security() -> dict[str, Any]:
    security = _security_manager()
    return {"policy": security.describe_policy()}


@router.get("/tool-audit")
async def list_tool_audit(
    limit: int = Query(50, ge=1, le=500),
    tool: str | None = Query(None),
    status: str | None = Query(None),
) -> dict[str, Any]:
    security = _security_manager()
    return {
        "items": security.audit.list_entries(limit=limit, tool=tool, status=status),
    }
