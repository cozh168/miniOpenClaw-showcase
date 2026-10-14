from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.chat import router as chat_router
from api.compress import router as compress_router
from api.config_api import router as config_router
from api.files import router as files_router
from api.sessions import router as sessions_router
from api.tool_security import router as tool_security_router
from api.tokens import router as tokens_router
from api.workbench import router as workbench_router
from config import get_settings, runtime_config
from graph.agent import agent_manager
from graph.checkpointer import init_checkpointer_async
from memory_module_v2.service.config import get_memory_backend
from service.knowledge_base import get_knowledge_base
from service.memory_indexer import memory_indexer
from service.http_security import configure_http_security
from tools.skills_scanner import refresh_snapshot


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    await init_checkpointer_async()
    # 这些组件都按 backend 内相对路径组织（workspace/、skills/、memory_module_v1/）
    refresh_snapshot(settings.backend_dir)
    agent_manager.initialize(settings.backend_dir)
    memory_indexer.configure(settings.backend_dir)
    if get_memory_backend() == "v1":
        memory_indexer.rebuild_index()
    if runtime_config.get_rag_mode():
        get_knowledge_base(settings.backend_dir).ensure_loaded()
    yield


app = FastAPI(
    title="Mini-OpenClaw API",
    version="0.1.0",
    lifespan=lifespan,
)

configure_http_security(app)

app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(sessions_router, prefix="/api", tags=["sessions"])
app.include_router(files_router, prefix="/api", tags=["files"])
app.include_router(tokens_router, prefix="/api", tags=["tokens"])
app.include_router(compress_router, prefix="/api", tags=["compress"])
app.include_router(config_router, prefix="/api", tags=["config"])
app.include_router(tool_security_router, prefix="/api", tags=["tool-security"])
app.include_router(workbench_router, prefix="/api", tags=["workbench"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
