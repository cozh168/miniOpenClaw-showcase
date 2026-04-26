from __future__ import annotations

from typing import Any

import pytest

from backend.graph.agent_factory import AgentConfig, create_agent_from_config


def test_guardian_middleware_before_summarization_when_both_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeSummarizationMiddleware:
        def __init__(self, **_kwargs):
            pass

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "backend.graph.agent_factory.SummarizationMiddleware",
        _FakeSummarizationMiddleware,
    )
    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=True,
        use_summarization=True,
    )
    create_agent_from_config(config)

    middleware = list(captured["middleware"])
    assert len(middleware) == 2
    assert middleware[0].__class__.__name__ == "GuardianMiddleware"
    assert middleware[1].__class__.__name__ == "_FakeSummarizationMiddleware"


def test_only_guardian_middleware_when_summarization_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=True,
        use_summarization=False,
    )
    create_agent_from_config(config)

    middleware = list(captured["middleware"])
    assert len(middleware) == 1
    assert middleware[0].__class__.__name__ == "GuardianMiddleware"


def test_no_guardian_middleware_when_disabled_and_no_summarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=False,
        use_summarization=False,
    )
    create_agent_from_config(config)

    assert captured["middleware"] == ()


def test_only_summarization_when_guardian_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeSummarizationMiddleware:
        def __init__(self, **_kwargs):
            pass

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "backend.graph.agent_factory.SummarizationMiddleware",
        _FakeSummarizationMiddleware,
    )
    monkeypatch.setattr("backend.graph.agent_factory.create_agent", _fake_create_agent)

    config = AgentConfig(
        llm=object(),
        tools=[],
        system_prompt="",
        guardian_enabled=False,
        use_summarization=True,
    )
    create_agent_from_config(config)

    middleware = list(captured["middleware"])
    assert len(middleware) == 1
    assert middleware[0].__class__.__name__ == "_FakeSummarizationMiddleware"
