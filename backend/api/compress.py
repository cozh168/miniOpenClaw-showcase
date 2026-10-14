from __future__ import annotations

from fastapi import APIRouter, HTTPException

from graph.agent import agent_manager
from graph.checkpointer import get_checkpointer
from service.session_manager import SessionBusyError

router = APIRouter()


@router.post("/sessions/{session_id}/compress")
async def compress_session(session_id: str) -> dict[str, int]:
    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    try:
        token = session_manager.begin_chat(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        record = session_manager.get_history(session_id)
        messages = record.get("messages", [])
        if len(messages) < 4:
            raise HTTPException(status_code=400, detail="At least 4 messages are required")
        n_messages = max(4, len(messages) // 2)
        try:
            summary = await agent_manager.summarize_history(messages[:n_messages])
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Unable to summarize the conversation; original history was preserved") from exc
        if not summary.strip():
            raise HTTPException(status_code=502, detail="The model returned an empty summary; original history was preserved")
        # Next turn must rebuild from the persisted summary instead of retaining
        # the uncompressed checkpoint's full original conversation. Clear first:
        # a checkpoint failure must not publish a partially completed compression.
        # If the later file write fails, the complete JSON history still restores.
        try:
            await get_checkpointer().adelete_thread(session_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Could not reset conversation checkpoint; original history was preserved") from exc
        return session_manager.compress_history(session_id, summary, n_messages)
    finally:
        session_manager.end_chat(session_id, token)
