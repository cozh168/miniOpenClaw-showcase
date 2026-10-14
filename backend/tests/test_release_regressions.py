from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.chat import ChatRequest, chat, router as chat_router
from api.compress import compress_session, router as compress_router
from graph.agent import AgentManager, agent_manager
from graph.context import RequestContext
from service.http_security import allowed_browser_origins, configure_http_security
from service.session_manager import SessionBusyError, SessionManager
from tools.skills_scanner import SkillRecord, build_snapshot, refresh_snapshot


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager = SessionManager(tmp_path)
    monkeypatch.setattr(agent_manager, "base_dir", tmp_path)
    monkeypatch.setattr(agent_manager, "session_manager", manager)
    monkeypatch.setattr("api.chat.get_memory_backend", lambda: "off")
    return manager


def _security_app():
    app = FastAPI()
    configure_http_security(app)
    invoked = []

    @app.api_route("/api/action", methods=["GET", "POST", "DELETE"])
    async def action():
        invoked.append(True)
        return {"ok": True}

    return app, invoked


@pytest.mark.parametrize("origin", ["https://external.example", "null", "http://localhost:7788.attacker.example"])
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_untrusted_browser_requests_are_rejected_before_the_action(origin, method):
    app, invoked = _security_app()
    with TestClient(app) as client:
        response = client.request(method, "/api/action", headers={"Origin": origin})
    assert response.status_code == 403
    assert invoked == []
    assert "access-control-allow-origin" not in response.headers


def test_trusted_origins_preflight_and_cli_requests_work():
    app, invoked = _security_app()
    with TestClient(app) as client:
        response = client.options("/api/action", headers={
            "Origin": "http://127.0.0.1:7788",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:7788"
        assert client.post("/api/action", headers={"Origin": "http://localhost:7788"}).status_code == 200
        assert client.post("/api/action").status_code == 200
        assert client.options("/api/action", headers={"Origin": "https://external.example", "Access-Control-Request-Method": "POST"}).status_code == 403
    assert len(invoked) == 2


def test_explicit_additional_origins(monkeypatch):
    monkeypatch.setenv("BACKEND_ALLOWED_ORIGINS", "https://claw.example.com,http://192.168.1.2:7788")
    origins = allowed_browser_origins()
    assert "https://claw.example.com" in origins
    assert "http://192.168.1.2:7788" in origins
    monkeypatch.setenv("BACKEND_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError):
        allowed_browser_origins()
    monkeypatch.setenv("BACKEND_ALLOWED_ORIGINS", "https://example.com/path")
    with pytest.raises(ValueError):
        allowed_browser_origins()


def test_failed_session_publish_preserves_full_original(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    manager.save_message(session_id, "user", "Keep this history")
    path = manager._session_path(session_id)
    original = path.read_bytes()

    def fail_publish(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("service.session_manager.os.replace", fail_publish)
    with pytest.raises(OSError):
        manager.save_exchange(session_id, "Must not be half saved", [{"content": "response", "tool_calls": []}])
    assert path.read_bytes() == original
    assert list(manager.sessions_dir.glob(".session-*")) == []


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_failed_trash_publication_preserves_source_and_can_be_retried(manager, monkeypatch, operation):
    session_id = manager.create_session()["id"]
    manager.save_message(session_id, "user", "Preserve this")
    if operation == "restore":
        manager.delete_session(session_id)
    source = manager._session_path(session_id) if operation == "delete" else manager._trash_path(session_id)
    target = manager._trash_path(session_id) if operation == "delete" else manager._session_path(session_id)
    original = source.read_bytes()

    with monkeypatch.context() as patch:
        def fail_publish(*args, **kwargs):
            raise OSError("simulated disk failure")
        patch.setattr("service.session_manager.os.link", fail_publish)
        with pytest.raises(OSError):
            getattr(manager, f"{operation}_session")(session_id)
    assert source.read_bytes() == original
    assert not target.exists()
    assert not list(source.parent.glob(".session-*"))
    getattr(manager, f"{operation}_session")(session_id)
    assert target.exists()
    assert not source.exists()


def test_archive_names_do_not_collide_for_two_compressions_in_the_same_second(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    for number in range(8):
        manager.save_message(session_id, "user", f"message {number}")
    monkeypatch.setattr("service.session_manager.time.time", lambda: 1234)
    manager.compress_history(session_id, "first summary", 2)
    manager.compress_history(session_id, "second summary", 2)
    archives = list(manager.archive_dir.glob("*.json"))
    assert len(archives) == 2
    assert {json.loads(path.read_text())["messages"][0]["content"] for path in archives} == {"message 0", "message 2"}


def test_filename_identity_cannot_overwrite_a_different_session(manager):
    first = manager.create_session()
    second = manager.create_session()
    record = {**first, "id": second["id"]}
    manager._session_path(first["id"]).write_text(json.dumps(record))
    manager.rename_session(first["id"], "Renamed first")
    assert manager.get_history(first["id"])["id"] == first["id"]
    assert manager.get_history(second["id"])["title"] == second["title"]


def test_reading_unknown_session_never_creates_a_file(manager):
    for operation in (manager.load_session, manager.get_history, manager.rename_session):
        args = ("unknown", "title") if operation == manager.rename_session else ("unknown",)
        with pytest.raises(FileNotFoundError):
            operation(*args)
    assert manager.list_sessions() == []


def test_compression_holds_lease_and_clears_checkpoint(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    for index in range(4):
        manager.save_message(session_id, "user" if index % 2 == 0 else "assistant", f"message {index}")
    deleted_threads = []

    async def summarize(messages):
        assert len(messages) == 4
        with pytest.raises(SessionBusyError):
            manager.begin_chat(session_id)
        with pytest.raises(SessionBusyError):
            manager.delete_session(session_id)
        return "A useful summary"

    async def delete_thread(value):
        deleted_threads.append(value)

    monkeypatch.setattr(agent_manager, "summarize_history", summarize)
    monkeypatch.setattr("api.compress.get_checkpointer", lambda: SimpleNamespace(adelete_thread=delete_thread))
    assert asyncio.run(compress_session(session_id)) == {"archived_count": 4, "remaining_count": 0}
    assert deleted_threads == [session_id]
    assert manager.load_session_for_agent(session_id)[0]["content"].endswith("A useful summary")
    token = manager.begin_chat(session_id)
    manager.end_chat(session_id, token)


def test_compression_rejects_unknown_busy_and_failed_summary_releases_lease(manager, monkeypatch):
    app = FastAPI()
    app.include_router(compress_router, prefix="/api")
    with TestClient(app) as client:
        assert client.post("/api/sessions/unknown/compress").status_code == 404
    assert manager.list_sessions() == []
    session_id = manager.create_session()["id"]
    token = manager.begin_chat(session_id)
    with pytest.raises(HTTPException) as error:
        asyncio.run(compress_session(session_id))
    assert error.value.status_code == 409
    manager.end_chat(session_id, token)
    for index in range(4):
        manager.save_message(session_id, "user", f"message {index}")

    async def fail(messages):
        raise RuntimeError("model failed")
    monkeypatch.setattr(agent_manager, "summarize_history", fail)
    with pytest.raises(HTTPException) as error:
        asyncio.run(compress_session(session_id))
    assert error.value.status_code == 502
    assert len(manager.load_session(session_id)) == 4
    token = manager.begin_chat(session_id)
    manager.end_chat(session_id, token)


def test_parallel_tool_results_have_correct_ids_and_one_response_boundary(tmp_path, monkeypatch):
    class Graph:
        async def astream(self, *args, **kwargs):
            yield "updates", {"model": {"messages": [SimpleNamespace(type="ai", content="", tool_calls=[
                {"id": "call-one", "name": "python_repl", "args": {"code": "print(1)"}},
                {"id": "call-two", "name": "python_repl", "args": {"code": "print(2)"}},
            ])]}}
            # Results are intentionally emitted in the opposite order.
            for call_id, output in [("call-two", "2"), ("call-one", "1")]:
                yield "updates", {"tools": {"messages": [SimpleNamespace(type="tool", tool_calls=[], tool_call_id=call_id, name="python_repl", content=output)]}}
            yield "messages", (SimpleNamespace(content="Both complete"), {"langgraph_node": "model"})
            yield "updates", {"model": {"messages": [SimpleNamespace(type="ai", content="Both complete", tool_calls=[])]}}

    manager = AgentManager()
    manager.base_dir = tmp_path
    monkeypatch.setattr(manager, "_build_agent", lambda: Graph())
    monkeypatch.setattr("graph.agent.get_memory_backend", lambda: "off")
    monkeypatch.setattr("graph.agent.runtime_config.get_rag_mode", lambda: False)

    async def run():
        return [event async for event in manager.astream("Run both", [], RequestContext(thread_id="parallel"))]
    events = asyncio.run(run())
    starts = [event for event in events if event["type"] == "tool_start"]
    assert [event["call_id"] for event in starts] == ["call-one", "call-two"]
    ends = [event for event in events if event["type"] == "tool_end"]
    assert [(event["call_id"], event["output"]) for event in ends] == [("call-two", "2"), ("call-one", "1")]
    assert sum(event["type"] == "new_response" for event in events) == 1
    assert next(index for index, event in enumerate(events) if event["type"] == "new_response") > max(index for index, event in enumerate(events) if event["type"] == "tool_end")
    assert events[-1] == {"type": "done", "content": "Both complete"}


def test_parallel_tool_outputs_persist_on_the_right_calls(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    manager.save_message(session_id, "user", "Earlier message skips title generation")

    async def stream(*args, **kwargs):
        yield {"type": "tool_start", "call_id": "one", "tool": "python_repl", "input": "print(1)"}
        yield {"type": "tool_start", "call_id": "two", "tool": "python_repl", "input": "print(2)"}
        yield {"type": "tool_end", "call_id": "two", "tool": "python_repl", "output": "2"}
        yield {"type": "tool_end", "call_id": "one", "tool": "python_repl", "output": "1"}
        yield {"type": "new_response"}
        yield {"type": "token", "content": "Completed"}
        yield {"type": "done", "content": "Completed"}

    monkeypatch.setattr(agent_manager, "astream", stream)
    app = FastAPI()
    app.include_router(chat_router, prefix="/api")
    with TestClient(app) as client:
        assert client.post("/api/chat", json={"session_id": session_id, "message": "Run both"}).status_code == 200
    persisted = manager.load_session(session_id)
    assert persisted[-1]["content"] == "Completed"
    calls = persisted[-2]["tool_calls"]
    assert [(call["call_id"], call["output"]) for call in calls] == [("one", "1"), ("two", "2")]


def test_nonstream_chat_returns_text_and_reports_model_errors(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    manager.save_message(session_id, "user", "Already named")

    async def stream(*args, **kwargs):
        yield {"type": "done", "content": "Plain result"}
    monkeypatch.setattr(agent_manager, "astream", stream)
    app = FastAPI()
    app.include_router(chat_router, prefix="/api")
    with TestClient(app) as client:
        assert client.post("/api/chat", json={"session_id": session_id, "message": "test", "stream": False}).json() == {"content": "Plain result"}

        async def fail(*args, **kwargs):
            yield {"type": "token", "content": "partial"}
            raise RuntimeError("simulated model failure")
        monkeypatch.setattr(agent_manager, "astream", fail)
        response = client.post("/api/chat", json={"session_id": session_id, "message": "test", "stream": False})
        assert response.status_code == 502
        assert response.json()["detail"] == "simulated model failure"
    token = manager.begin_chat(session_id)
    manager.end_chat(session_id, token)


def test_skills_snapshot_escapes_user_text_as_valid_xml():
    source = SkillRecord(name='Research & "analysis"', description="Compare x < y & keep > 0", path="skills/research/SKILL.md")
    root = ElementTree.fromstring(build_snapshot([source]))
    skill = root.find("skill")
    assert skill.attrib["name"] == source.name
    assert skill.find("description").text == source.description


def test_skill_snapshot_never_follows_a_symbolic_target(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("Do not overwrite")
    (skills / "SKILLS_SNAPSHOT.md").symlink_to(outside)
    with pytest.raises(ValueError):
        refresh_snapshot(tmp_path)
    assert outside.read_text() == "Do not overwrite"


@pytest.mark.parametrize("backend", ["off", "v2", "v1"])
def test_startup_only_rebuilds_v1_memory_when_it_is_enabled(tmp_path, monkeypatch, backend):
    import app as application
    calls = []

    async def init_checkpoint():
        return None
    monkeypatch.setattr(application, "get_settings", lambda: SimpleNamespace(backend_dir=tmp_path))
    monkeypatch.setattr(application, "init_checkpointer_async", init_checkpoint)
    monkeypatch.setattr(application, "refresh_snapshot", lambda base: None)
    monkeypatch.setattr(application.agent_manager, "initialize", lambda base: None)
    monkeypatch.setattr(application.memory_indexer, "configure", lambda base: calls.append("configure"))
    monkeypatch.setattr(application.memory_indexer, "rebuild_index", lambda: calls.append("rebuild"))
    monkeypatch.setattr(application.runtime_config, "get_rag_mode", lambda: False)
    monkeypatch.setattr(application, "get_memory_backend", lambda: backend)

    async def run():
        async with application.lifespan(application.app):
            pass
    asyncio.run(run())
    assert calls == (["configure", "rebuild"] if backend == "v1" else ["configure"])


@pytest.mark.parametrize("backend", ["off", "v2", "v1"])
def test_editing_old_memory_only_rebuilds_enabled_v1_index(monkeypatch, backend):
    from api import files
    calls = []
    monkeypatch.setattr(files, "get_memory_backend", lambda: backend)
    monkeypatch.setattr(files.memory_indexer, "rebuild_index", lambda: calls.append(True))
    files.notify_file_updated("memory_module_v1/long_term_memory/MEMORY.md")
    assert calls == ([True] if backend == "v1" else [])


def test_failed_checkpoint_reset_preserves_original_history(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    for number in range(4):
        manager.save_message(session_id, "user", f"message {number}")
    original = manager.load_session(session_id)

    async def summarize(messages):
        return "New summary"
    async def fail(value):
        raise OSError("checkpoint unavailable")
    monkeypatch.setattr(agent_manager, "summarize_history", summarize)
    monkeypatch.setattr("api.compress.get_checkpointer", lambda: SimpleNamespace(adelete_thread=fail))
    with pytest.raises(HTTPException) as error:
        asyncio.run(compress_session(session_id))
    assert error.value.status_code == 503
    assert manager.load_session(session_id) == original
    assert list(manager.archive_dir.glob("*.json")) == []
    token = manager.begin_chat(session_id)
    manager.end_chat(session_id, token)


def test_empty_summary_cannot_destroy_conversation(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    for number in range(4):
        manager.save_message(session_id, "user", f"message {number}")

    async def summarize(messages):
        return ""
    monkeypatch.setattr(agent_manager, "summarize_history", summarize)
    with pytest.raises(HTTPException) as error:
        asyncio.run(compress_session(session_id))
    assert error.value.status_code == 502
    assert len(manager.load_session(session_id)) == 4
    assert list(manager.archive_dir.glob("*.json")) == []


def test_session_storage_directories_cannot_be_symbolic(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "memory_module_v1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        SessionManager(tmp_path)
    assert list(outside.iterdir()) == []


def test_nonstream_error_releases_lease_before_event_loop_shutdown(manager, monkeypatch):
    session_id = manager.create_session()["id"]

    async def fail(*args, **kwargs):
        yield {"type": "token", "content": "partial"}
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(agent_manager, "astream", fail)

    async def run():
        with pytest.raises(HTTPException) as error:
            await chat(ChatRequest(session_id=session_id, message="test", stream=False))
        assert error.value.status_code == 502
        # This check runs in the same still-active loop, before asyncio.run()
        # shuts down abandoned generators and could otherwise mask a lease leak.
        token = manager.begin_chat(session_id)
        manager.end_chat(session_id, token)
    asyncio.run(run())


def test_app_import_loads_explicit_origins_from_dotenv(tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    backend = Path(__file__).resolve().parents[1]
    config_copy = tmp_path / "config"
    config_copy.mkdir()
    for filename in ("__init__.py", "config.py"):
        shutil.copyfile(backend / "config" / filename, config_copy / filename)
    (config_copy / ".env").write_text("BACKEND_ALLOWED_ORIGINS=https://claw.example.test\n")
    env = dict(os.environ)
    env.pop("BACKEND_ALLOWED_ORIGINS", None)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(backend)
    result = subprocess.run([
        sys.executable, "-c",
        "import app; from service.http_security import allowed_browser_origins; assert 'https://claw.example.test' in allowed_browser_origins()",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_token_count_rejects_unknown_and_deleted_sessions(manager):
    from api.tokens import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        assert client.get("/api/tokens/session/unknown").status_code == 404
        session_id = manager.create_session()["id"]
        manager.delete_session(session_id)
        assert client.get(f"/api/tokens/session/{session_id}").status_code == 404
    assert manager.list_sessions() == []


def test_file_token_count_uses_file_whitelist_and_text_validation(manager, tmp_path, monkeypatch):
    from api import tokens
    app = FastAPI()
    app.include_router(tokens.router, prefix="/api")
    monkeypatch.setattr(tokens, "_count_tokens", lambda text: len(text))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.md").write_text("example")
    (workspace / "binary.md").write_bytes(b"\xff\xfe")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside the allowed root")
    (workspace / "linked.md").symlink_to(outside)
    with TestClient(app) as client:
        result = client.post("/api/tokens/files", json={"paths": ["workspace/note.md", "workspace/missing.md"]})
        assert result.status_code == 200
        assert result.json()["total_tokens"] == len("example")
        for path in ("../outside.md", "outside.md", "workspace/linked.md", "workspace/binary.md"):
            assert client.post("/api/tokens/files", json={"paths": [path]}).status_code == 400


def test_literal_special_tokens_are_counted_as_ordinary_text(monkeypatch):
    from api import tokens
    calls = []
    def encode(text, *, disallowed_special):
        assert disallowed_special == ()
        calls.append(text)
        return [1, 2, 3]
    monkeypatch.setattr(tokens, "_get_encoder", lambda: SimpleNamespace(encode=encode))
    assert tokens._count_tokens("Example <|endoftext|> marker") == 3
    assert calls == ["Example <|endoftext|> marker"]


class _SummaryModel:
    def __init__(self, response):
        self.response = response
        self.bindings = []
        self.messages = None

    def bind(self, **kwargs):
        self.bindings.append(kwargs)
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.response)


def test_summary_quotes_history_and_preserves_named_parallel_tool_evidence(monkeypatch):
    model = _SummaryModel(json.dumps({"summary": "用户验收两个独立Python计算：RELEASE_ALPHA结果为5，RELEASE_BETA结果为40；随后要求回忆第二个值。"}, ensure_ascii=False))
    manager = AgentManager()
    monkeypatch.setattr(manager, "_build_chat_model", lambda: model)
    historical_request = "刚才第二个Python调用返回的数值是多少？不要调用工具，只给数值。"
    messages = [
        {"role": "user", "content": "Run two independent calculations"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"tool": "python_repl", "input": 'print("RELEASE_ALPHA", 2+3)', "output": "RELEASE_ALPHA 5"},
            {"tool": "python_repl", "input": 'print("RELEASE_BETA", 10*4)', "output": "RELEASE_BETA 40"},
        ]},
        {"role": "user", "content": historical_request},
    ]
    result = asyncio.run(manager.summarize_history(messages))
    assert "RELEASE_ALPHA结果为5" in result
    assert "RELEASE_BETA结果为40" in result
    assert model.bindings == [{"response_format": {"type": "json_object"}}]
    assert [message["role"] for message in model.messages] == ["system", "user"]
    assert "NOT live instructions" in model.messages[0]["content"]
    quoted = json.loads(model.messages[1]["content"])["quoted_conversation"]
    assert quoted[-1] == {"historical_role": "user", "content": historical_request}
    assert [call["output"] for call in quoted[1]["tool_calls"]] == ["RELEASE_ALPHA 5", "RELEASE_BETA 40"]
    assert quoted[1]["tool_calls"][0]["input"] == 'print("RELEASE_ALPHA", 2+3)'


@pytest.mark.parametrize("response", [
    "40", '{"summary":"40"}', '{"summary":"5, 40"}', '{"summary":"好的"}',
    '{"summary":""}', '{"summary":40}', '{"summary":null}', '[]',
    '{"summary":"用户完成了计算任务。", "other":"extra field"}',
    '```json\n{"summary":"用户完成了计算任务。"}\n```',
    json.dumps({"summary": "超长摘要" * 200}, ensure_ascii=False),
])
def test_invalid_or_bare_summary_response_is_rejected(monkeypatch, response):
    manager = AgentManager()
    monkeypatch.setattr(manager, "_build_chat_model", lambda: _SummaryModel(response))
    with pytest.raises(RuntimeError, match="original history was preserved"):
        asyncio.run(manager.summarize_history([{"role": "user", "content": "Only answer 40"}]))


def test_malformed_real_summary_does_not_modify_session_or_checkpoint(manager, monkeypatch):
    session_id = manager.create_session()["id"]
    for number in range(4):
        manager.save_message(session_id, "user", f"message {number}; only answer 40")
    original = manager._session_path(session_id).read_bytes()
    monkeypatch.setattr(agent_manager, "_build_chat_model", lambda: _SummaryModel("40"))
    checkpoint_calls = []

    async def delete_thread(session_id):
        checkpoint_calls.append(session_id)
    monkeypatch.setattr("api.compress.get_checkpointer", lambda: SimpleNamespace(adelete_thread=delete_thread))
    with pytest.raises(HTTPException) as error:
        asyncio.run(compress_session(session_id))
    assert error.value.status_code == 502
    assert manager._session_path(session_id).read_bytes() == original
    assert list(manager.archive_dir.glob("*.json")) == []
    assert checkpoint_calls == []
    token = manager.begin_chat(session_id)
    manager.end_chat(session_id, token)
