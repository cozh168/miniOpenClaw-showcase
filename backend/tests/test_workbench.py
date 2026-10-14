from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.chat import ChatRequest, chat, router as chat_router
from api.files import notify_file_updated as real_notify_file_updated, router as files_router
from api.sessions import router as sessions_router
from api.workbench import router as workbench_router
from graph.agent import AgentManager, agent_manager
from graph.context import RequestContext
from service.session_manager import SessionBusyError, SessionManager
from service.workbench_files import MAX_TEXT_BYTES, list_workbench_files, write_text_file
from tools.skills_scanner import refresh_snapshot, scan_skills


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_manager, "base_dir", tmp_path)
    monkeypatch.setattr(agent_manager, "session_manager", SessionManager(tmp_path))
    monkeypatch.setattr(agent_manager, "tools", [])
    monkeypatch.setattr("api.files.notify_file_updated", lambda _: None)
    monkeypatch.setattr("api.workbench.notify_file_updated", lambda _: None)
    app = FastAPI()
    for router in (files_router, workbench_router, sessions_router, chat_router):
        app.include_router(router, prefix="/api")
    with TestClient(app) as test_client:
        yield test_client


def _flatten(nodes):
    for node in nodes:
        yield node
        yield from _flatten(node.get("children", []))


def test_file_tree_lists_real_files_and_excludes_sensitive_paths(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "z.md").write_text("# Hello", encoding="utf-8")
    (workspace / "a.pdf").write_bytes(b"%PDF binary")
    (workspace / "hidden.txt").write_bytes(b"\xff\xfe")
    (workspace / "folder").mkdir()
    (workspace / "folder" / "nested.py").write_text("print(1)")
    (workspace / ".env").write_text("API_KEY=sensitive")
    (workspace / "credentials.json").write_text('{}')
    (workspace / "linked.md").symlink_to(workspace / "z.md")
    response = client.get("/api/workbench/files")
    assert response.status_code == 200
    root = response.json()["files"][0]
    assert [child["name"] for child in root["children"]] == ["folder", "a.pdf", "hidden.txt", "z.md"]
    entries = {node["path"]: node for node in _flatten([root])}
    assert entries["workspace/z.md"]["editable"] is True
    assert entries["workspace/a.pdf"]["editable"] is False
    assert entries["workspace/hidden.txt"]["editable"] is False
    assert not any(".env" in name or "credentials" in name or "linked" in name for name in entries)


def test_tree_bounds_results_and_omits_symlink_roots(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(550):
        (workspace / f"{index:03}.md").write_text("x")
    nodes = list_workbench_files(tmp_path)
    assert len(list(_flatten(nodes))) == 500
    (tmp_path / "skills").symlink_to(workspace, target_is_directory=True)
    assert all(node["path"] != "skills" for node in list_workbench_files(tmp_path))


def test_large_folder_cannot_hide_other_workspace_roots(tmp_path):
    for root in ("knowledge", "workspace", "skills", "memory_module_v1/long_term_memory"):
        (tmp_path / root).mkdir(parents=True)
    for index in range(550):
        (tmp_path / "knowledge" / f"{index:03}.md").write_text("x")
    nodes = list_workbench_files(tmp_path)
    assert len(list(_flatten(nodes))) == 500
    assert {node["path"] for node in nodes} == {
        "knowledge", "workspace", "skills", "memory_module_v1/long_term_memory",
    }


@pytest.mark.parametrize("path", [
    "workspace/../config/.env", "workspace/sub/../../config/config.json",
    "/workspace/a.txt", "workspace/.env", "workspace/.hidden/note.md",
    "workspace/secret.txt", "workspace/api-key.json", "config/config.json",
    "workspace/bad:name.md", "workspace/../outside.md",
    "workspace/" + "x" * 256 + ".md", "workspace/" + "sub/" * 33 + "a.md",
])
def test_read_and_save_reject_unsafe_paths(client, path):
    assert client.get("/api/files", params={"path": path}).status_code == 400
    assert client.post("/api/files", json={"path": path, "content": "x"}).status_code == 400
    assert client.post("/api/workbench/files", json={"path": path, "content": "x"}).status_code == 400


def test_symlink_cannot_escape_editable_root(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "config"
    external.mkdir()
    target = external / "settings.md"
    target.write_text("keep me")
    (workspace / "linked").symlink_to(external, target_is_directory=True)
    path = "workspace/linked/settings.md"
    assert client.get("/api/files", params={"path": path}).status_code == 400
    assert client.post("/api/files", json={"path": path, "content": "new"}).status_code == 400
    assert target.read_text() == "keep me"


def test_create_and_edit_real_text_and_do_not_overwrite_on_create(client, tmp_path):
    payload = {"path": "workspace/项目/test.py", "content": "print('你好')\n"}
    response = client.post("/api/workbench/files", json=payload)
    assert response.status_code == 200
    assert response.json() == {"path": payload["path"], "ok": True}
    assert (tmp_path / payload["path"]).read_text() == payload["content"]
    assert client.post("/api/workbench/files", json={**payload, "content": "overwrite"}).status_code == 409
    assert client.get("/api/files", params={"path": payload["path"]}).json()["content"] == payload["content"]
    assert client.post("/api/files", json={**payload, "content": "print(2)"}).status_code == 200
    assert (tmp_path / payload["path"]).read_text() == "print(2)"


def test_binary_directory_and_large_text_fail_clearly_without_changes(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "folder").mkdir()
    (workspace / "sample.pdf").write_bytes(b"%PDF-1.7")
    (workspace / "invalid.md").write_bytes(b"\xff\xfe")
    (workspace / "big.txt").write_bytes(b"a" * (MAX_TEXT_BYTES + 1))
    for name, expected in (("folder", 400), ("sample.pdf", 400), ("invalid.md", 400), ("big.txt", 413)):
        path = f"workspace/{name}"
        assert client.get("/api/files", params={"path": path}).status_code == expected
        assert client.post("/api/files", json={"path": path, "content": "replace"}).status_code == expected
    oversized = {"path": "workspace/new.md", "content": "中" * (MAX_TEXT_BYTES // 3 + 1)}
    assert client.post("/api/workbench/files", json=oversized).status_code == 413
    assert not (workspace / "new.md").exists()
    assert (workspace / "sample.pdf").read_bytes() == b"%PDF-1.7"
    assert client.post("/api/files", json={"path": "workspace/null.txt", "content": "a\u0000b"}).status_code == 400


def test_runtime_summary_has_config_and_excludes_secrets(client, monkeypatch):
    settings = SimpleNamespace(llm_provider="deepseek", llm_model="deepseek-chat", llm_api_key="SECRET_API_KEY")
    security = SimpleNamespace(describe_policy=lambda: {"enabled_tools": {"python_repl": True, "terminal": False}})
    monkeypatch.setattr("api.workbench.get_settings", lambda: settings)
    monkeypatch.setattr("api.workbench.get_tool_security_manager", lambda _: security)
    monkeypatch.setattr("api.workbench.runtime_config.get_rag_mode", lambda: False)
    monkeypatch.setattr("api.workbench.get_memory_backend", lambda: "off")
    monkeypatch.setattr(agent_manager, "tools", [SimpleNamespace(name="search_knowledge_base")])
    response = client.get("/api/workbench/runtime")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "deepseek"
    assert data["model"] == "deepseek-chat"
    assert data["memory_backend"] == "off"
    assert data["rag_enabled"] is False
    assert {"name": "search_knowledge_base", "enabled": False} in data["tools"]
    assert any(tool == {"name": "python_repl", "enabled": True} for tool in data["tools"])
    assert "SECRET_API_KEY" not in response.text
    assert "api_key" not in response.text


def test_soft_delete_restore_preserves_messages_and_archive(client, tmp_path):
    session = client.post("/api/sessions", json={"title": "复利计算"}).json()
    session_id = session["id"]
    manager = agent_manager.session_manager
    manager.save_message(session_id, "user", "hello")
    manager.save_message(session_id, "assistant", "world")
    manager.compress_history(session_id, "greeting", 1)
    original = manager.load_session_record(session_id)
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.get("/api/sessions").json() == []
    trash_response = client.get("/api/sessions/trash")
    assert trash_response.status_code == 200
    assert trash_response.json()[0]["id"] == session_id
    assert trash_response.json()[0]["deleted_at"] > 0
    assert client.get(f"/api/sessions/{session_id}/history").status_code == 404
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404
    assert client.put(f"/api/sessions/{session_id}", json={"title": "resurrect"}).status_code == 404
    with pytest.raises(FileNotFoundError):
        manager.load_session_record(session_id)
    restored = client.post(f"/api/sessions/{session_id}/restore")
    assert restored.status_code == 200
    assert restored.json()["messages"] == original["messages"]
    assert restored.json()["compressed_context"] == "greeting"
    assert restored.json()["title"] == "复利计算"
    assert client.get("/api/sessions/trash").json() == []
    assert len(list(manager.archive_dir.glob("*.json"))) == 1


def test_restore_conflict_cannot_overwrite_live_session(client):
    session = client.post("/api/sessions", json={"title": "original"}).json()
    session_id = session["id"]
    manager = agent_manager.session_manager
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    existing = {**session, "title": "active copy"}
    manager._session_path(session_id).write_text(json.dumps(existing))
    response = client.post(f"/api/sessions/{session_id}/restore")
    assert response.status_code == 409
    assert manager.get_history(session_id)["title"] == "active copy"
    assert manager.list_trash()[0]["title"] == "original"
    assert client.post("/api/sessions/missing/restore").status_code == 404
    assert client.get("/api/sessions/missing/history").status_code == 404
    assert not manager.session_exists("missing")


@pytest.mark.parametrize("session_id", ["../outside", "", "bad.name", "id/child"])
def test_session_manager_rejects_path_ids(tmp_path, session_id):
    manager = SessionManager(tmp_path)
    with pytest.raises(ValueError):
        manager.load_session_record(session_id)
    with pytest.raises(ValueError):
        manager.restore_session(session_id)


def test_uninitialized_workbench_returns_503(client, monkeypatch):
    monkeypatch.setattr(agent_manager, "base_dir", None)
    assert client.get("/api/workbench/files").status_code == 503
    assert client.get("/api/workbench/runtime").status_code == 503


def test_active_chat_prevents_delete_and_conflicting_chat(client):
    session_id = client.post("/api/sessions", json={"title": "active"}).json()["id"]
    manager = agent_manager.session_manager
    token = manager.begin_chat(session_id)
    try:
        assert client.delete(f"/api/sessions/{session_id}").status_code == 409
        assert manager.session_exists(session_id)
        assert manager.list_trash() == []
        with pytest.raises(SessionBusyError):
            manager.begin_chat(session_id)
        manager.save_message(session_id, "assistant", "completed normally")
    finally:
        manager.end_chat(session_id, token)
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    restored = client.post(f"/api/sessions/{session_id}/restore")
    assert restored.json()["id"] == session_id
    assert restored.json()["messages"][0]["content"] == "completed normally"


def test_loading_unknown_session_does_not_create_it(tmp_path):
    manager = SessionManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load_session_record("unknown")
    with pytest.raises(FileNotFoundError):
        manager.begin_chat("unknown")
    assert manager.list_sessions() == []
    assert not manager._session_path("unknown").exists()


def test_create_only_is_atomic_under_competing_writers(tmp_path):
    target = tmp_path / "workspace" / "race.md"

    def create(content):
        try:
            write_text_file(target, content, create_only=True)
            return (200, content)
        except HTTPException as exc:
            return (exc.status_code, content)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, ["first writer", "second writer"]))
    assert sorted(status for status, _ in results) == [200, 409]
    winner = next(content for status, content in results if status == 200)
    assert target.read_text() == winner
    assert list(target.parent.glob(".workbench-*")) == []


def test_failed_save_keeps_original_and_cleans_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "original.md"
    target.write_text("original content")

    def fail_publish(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("service.workbench_files.os.replace", fail_publish)
    with pytest.raises(HTTPException) as exc_info:
        write_text_file(target, "new content")
    assert exc_info.value.status_code == 400
    assert target.read_text() == "original content"
    assert list(tmp_path.glob(".workbench-*")) == []


@pytest.mark.parametrize("title_fails", [False, True])
def test_chat_completion_releases_lease_and_stale_background_cannot_release_next_chat(client, monkeypatch, title_fails):
    manager = agent_manager.session_manager
    session_id = manager.create_session()["id"]

    async def fake_stream(*args, **kwargs):
        yield {"type": "token", "content": "Test response"}
        yield {"type": "done", "content": "Test response"}

    async def fake_title(*args, **kwargs):
        if title_fails:
            raise RuntimeError("simulated title failure")
        return "Test chat"

    monkeypatch.setattr(agent_manager, "astream", fake_stream)
    monkeypatch.setattr(agent_manager, "generate_title", fake_title)
    monkeypatch.setattr("api.chat.get_memory_backend", lambda: "off")

    async def run():
        response = await chat(ChatRequest(session_id=session_id, message="Test request"))
        with pytest.raises(HTTPException) as exc_info:
            await chat(ChatRequest(session_id=session_id, message="Competing request"))
        assert exc_info.value.status_code == 409
        events = [event async for event in response.body_iterator]
        assert any(event.startswith("event: done") for event in events)
        assert not any(event.startswith("event: error") for event in events)
        assert [message["role"] for message in manager.load_session(session_id)] == ["user", "assistant"]
        next_token = manager.begin_chat(session_id)
        try:
            await response.background()
            with pytest.raises(SessionBusyError):
                manager.begin_chat(session_id)
        finally:
            manager.end_chat(session_id, next_token)

    asyncio.run(run())


def test_chat_stream_error_releases_lease(client, monkeypatch):
    manager = agent_manager.session_manager
    session_id = manager.create_session()["id"]

    async def fake_stream(*args, **kwargs):
        yield {"type": "token", "content": "partial"}
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(agent_manager, "astream", fake_stream)

    async def run():
        response = await chat(ChatRequest(session_id=session_id, message="Test request"))
        events = [event async for event in response.body_iterator]
        assert any(event.startswith("event: error") for event in events)
        token = manager.begin_chat(session_id)
        manager.end_chat(session_id, token)
        await response.background()

    asyncio.run(run())


def test_chat_cancellation_and_unstarted_stream_release_lease(client, monkeypatch):
    manager = agent_manager.session_manager
    session_id = manager.create_session()["id"]

    async def fake_stream(*args, **kwargs):
        yield {"type": "token", "content": "partial"}
        await asyncio.Event().wait()

    monkeypatch.setattr(agent_manager, "astream", fake_stream)

    async def run():
        response = await chat(ChatRequest(session_id=session_id, message="Test request"))
        assert (await anext(response.body_iterator)).startswith("event: token")
        pending = asyncio.create_task(anext(response.body_iterator))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await response.body_iterator.aclose()
        token = manager.begin_chat(session_id)
        manager.end_chat(session_id, token)
        await response.background()
        unstarted = await chat(ChatRequest(session_id=session_id, message="Another request"))
        with pytest.raises(SessionBusyError):
            manager.begin_chat(session_id)
        await unstarted.background()
        token = manager.begin_chat(session_id)
        manager.end_chat(session_id, token)

    asyncio.run(run())


def test_chat_rejects_unknown_and_deleted_sessions_without_creation(client):
    manager = agent_manager.session_manager
    assert client.post("/api/chat", json={"session_id": "unknown", "message": "hello"}).status_code == 404
    assert manager.list_sessions() == []
    session_id = manager.create_session()["id"]
    manager.delete_session(session_id)
    assert client.post("/api/chat", json={"session_id": session_id, "message": "hello"}).status_code == 404
    assert manager.list_sessions() == []


class _FakeCheckpointGraph:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.inputs = []

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.messages})

    async def astream(self, payload, **kwargs):
        self.inputs.append(payload["messages"])
        self.messages.extend(payload["messages"])
        self.messages.append({"role": "assistant", "content": "Context restored"})
        yield "updates", {"model": {"messages": [SimpleNamespace(
            type="ai", content="Context restored", tool_calls=[],
        )]}}


def test_empty_checkpoint_restores_json_history_once_after_restart(tmp_path, monkeypatch):
    original = SessionManager(tmp_path)
    session_id = original.create_session()["id"]
    original.save_message(session_id, "user", "My project is called PaperHelper")
    original.save_message(session_id, "assistant", "I will remember the project name")
    # A fresh manager simulates restarting with persisted JSON and no checkpoints.
    restored_manager = SessionManager(tmp_path)
    history = restored_manager.load_session_for_agent(session_id)
    graph = _FakeCheckpointGraph()
    manager = AgentManager()
    manager.base_dir = tmp_path
    manager.session_manager = restored_manager
    monkeypatch.setattr(manager, "_build_agent", lambda: graph)
    monkeypatch.setattr("graph.agent.get_memory_backend", lambda: "off")
    monkeypatch.setattr("graph.agent.runtime_config.get_rag_mode", lambda: False)

    async def run():
        context = RequestContext(thread_id=session_id)
        first = [event async for event in manager.astream("What is my project called?", history, context)]
        assert first[-1]["type"] == "done"
        assert graph.inputs[0] == history + [{"role": "user", "content": "What is my project called?"}]
        awaitable = manager.astream("Continue with its design", history, context)
        second = [event async for event in awaitable]
        assert second[-1]["type"] == "done"
        assert graph.inputs[1] == [{"role": "user", "content": "Continue with its design"}]
        assert sum(message["content"] == "My project is called PaperHelper" for message in graph.messages) == 1

    asyncio.run(run())


def test_existing_checkpoint_does_not_repeat_json_history(tmp_path, monkeypatch):
    graph = _FakeCheckpointGraph([{"role": "user", "content": "Stored checkpoint history"}])
    manager = AgentManager()
    manager.base_dir = tmp_path
    monkeypatch.setattr(manager, "_build_agent", lambda: graph)
    monkeypatch.setattr("graph.agent.get_memory_backend", lambda: "off")
    monkeypatch.setattr("graph.agent.runtime_config.get_rag_mode", lambda: False)
    history = [{"role": "user", "content": "Duplicate JSON history"}]

    async def run():
        _ = [event async for event in manager.astream("Next turn", history, RequestContext(thread_id="known"))]
        assert graph.inputs == [[{"role": "user", "content": "Next turn"}]]
        assert not any(item["content"] == "Duplicate JSON history" for item in graph.messages)

    asyncio.run(run())


def test_chat_passes_persisted_summary_and_history_for_checkpoint_restore(client, monkeypatch):
    manager = agent_manager.session_manager
    session_id = manager.create_session()["id"]
    manager.save_message(session_id, "user", "The principal is 10000 yuan")
    manager.save_message(session_id, "assistant", "Noted")
    manager.save_message(session_id, "user", "Use 5 percent growth")
    manager.compress_history(session_id, "The user's principal is 10000 yuan", 2)
    captured = []

    async def fake_stream(message, history, **kwargs):
        captured.extend(history)
        yield {"type": "done", "content": "Result"}

    monkeypatch.setattr(agent_manager, "astream", fake_stream)
    monkeypatch.setattr("api.chat.get_memory_backend", lambda: "off")
    response = client.post("/api/chat", json={"session_id": session_id, "message": "Calculate it"})
    assert response.status_code == 200
    assert captured[0]["content"].startswith("[Previous conversation summary]")
    assert "10000 yuan" in captured[0]["content"]
    assert captured[-1] == {"role": "user", "content": "Use 5 percent growth"}


def test_new_memory_file_is_visible_and_editable_in_tree(client):
    path = "memory/notes.md"
    assert client.post("/api/workbench/files", json={"path": path, "content": "User notes"}).status_code == 200
    nodes = client.get("/api/workbench/files").json()["files"]
    assert any(node["path"] == "memory" for node in nodes)
    file = next(node for node in _flatten(nodes) if node["path"] == path)
    assert file["editable"] is True
    assert client.get("/api/files", params={"path": path}).json()["content"] == "User notes"


@pytest.mark.parametrize("invalid_content", [
    "---\nname: [broken\n---\n# Body",
    "---\n- name\n- description\n---\n# Body",
    "---\nscalar value\n---\n# Body",
    "---\n---\n# Body",
    "---\nname: Missing separator\n# Body",
])
def test_invalid_skill_frontmatter_cannot_replace_or_create_a_file(client, tmp_path, invalid_content):
    path = "skills/example/SKILL.md"
    original = "---\nname: Example\ndescription: A valid skill\n---\n# Body\n"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(original)
    response = client.post("/api/files", json={"path": path, "content": invalid_content})
    assert response.status_code == 400
    assert "frontmatter" in response.json()["detail"]
    assert target.read_text() == original
    new_path = "skills/new-skill/SKILL.md"
    assert client.post("/api/workbench/files", json={"path": new_path, "content": invalid_content}).status_code == 400
    assert not (tmp_path / new_path).exists()


def test_bad_existing_skills_do_not_break_listing_or_snapshot_refresh(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.files.notify_file_updated", real_notify_file_updated)
    monkeypatch.setattr("api.workbench.notify_file_updated", real_notify_file_updated)
    skills = tmp_path / "skills"
    bad_contents = {
        "bad-yaml": b"---\nname: [invalid\n---\n",
        "bad-shape": b"---\n- name\n---\n",
        "bad-encoding": b"\xff\xfe",
        "bad-separator": b"---\nname: Missing separator",
    }
    for folder, content in bad_contents.items():
        directory = skills / folder
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_bytes(content)
    valid = "\ufeff---\r\nname: Useful Skill\r\ndescription: Valid Windows lines\r\n---\r\n# Instructions\r\n"
    response = client.post("/api/workbench/files", json={"path": "skills/good/SKILL.md", "content": valid})
    assert response.status_code == 200
    listing = client.get("/api/skills")
    assert listing.status_code == 200
    assert [record["name"] for record in listing.json()] == ["Useful Skill"]
    snapshot = (skills / "SKILLS_SNAPSHOT.md").read_text()
    assert "Useful Skill" in snapshot
    assert "bad-yaml" not in snapshot
    assert [skill.name for skill in scan_skills(tmp_path)] == ["Useful Skill"]
    assert refresh_snapshot(tmp_path).exists()
    assert client.post("/api/files", json={"path": "skills/good/SKILL.md", "content": "---\nname: Updated Skill\n---\n# Revised"}).status_code == 200
    assert "Updated Skill" in (skills / "SKILLS_SNAPSHOT.md").read_text()


def test_plain_markdown_skill_without_frontmatter_remains_compatible(client):
    response = client.post("/api/workbench/files", json={"path": "skills/plain/SKILL.md", "content": "# Legacy plain markdown skill"})
    assert response.status_code == 200
    assert client.get("/api/skills").json()[0]["name"] == "plain"


def test_yaml_description_can_contain_an_indented_separator(client):
    content = "---\nname: YAML Blocks\ndescription: |\n  ---\n  This is part of the description\n---\n# Body"
    assert client.post("/api/workbench/files", json={"path": "skills/blocks/SKILL.md", "content": content}).status_code == 200
    listing = client.get("/api/skills").json()
    assert listing[0]["description"] == "---\nThis is part of the description"
