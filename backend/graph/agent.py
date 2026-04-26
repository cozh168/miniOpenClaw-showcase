from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import get_settings, runtime_config
from graph.agent_factory import build_agent_config, create_agent_from_config
from graph.context import RequestContext
from graph.llm import build_llm_config_from_settings, get_llm
from memory_module_v2.integrations.middleware import build_memory_context
from memory_module_v2.service.config import get_memory_backend, get_memory_v2_inject_mode
from service.knowledge_base import get_knowledge_base
from service.memory_indexer import memory_indexer
from service.session_manager import SessionManager
from tools import get_all_tools

logger = logging.getLogger(__name__)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


class AgentManager:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self.session_manager: SessionManager | None = None
        self.tools = []

    def initialize(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.session_manager = SessionManager(base_dir)
        self.tools = get_all_tools(base_dir)

    def _build_chat_model(self):
        settings = get_settings()
        llm_config = build_llm_config_from_settings(settings, temperature=0.0, streaming=False)
        return get_llm(llm_config)

    def _build_agent(self):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")
        self.tools = get_all_tools(self.base_dir)
        config = build_agent_config(
            self.base_dir,
            self.tools,
            use_checkpointer=True,
        )
        return create_agent_from_config(config)

    def _build_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": str(item.get("content", ""))})
        return messages

    def _format_retrieval_context(self, results: list[dict[str, Any]]) -> str:
        lines = ["[RAG retrieved memory context]"]
        for idx, item in enumerate(results, start=1):
            text = str(item.get("text", "")).strip()
            source = str(item.get("source", "memory_module_v1/long_term_memory/MEMORY.md"))
            lines.append(f"{idx}. Source: {source}\n{text}")
        return "\n\n".join(lines)

    async def astream(
        self,
        message: str,
        history: list[dict[str, Any]],
        context: RequestContext | None = None,
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        memory_backend = get_memory_backend()
        turn_messages: list[dict[str, str]] = []

        if memory_backend == "v1":
            retrievals = memory_indexer.retrieve(message, top_k=3)
            yield {"type": "retrieval", "query": message, "results": retrievals}
            if retrievals:
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_retrieval_context(retrievals),
                    }
                )
        elif memory_backend == "v2" and get_memory_v2_inject_mode() == "always":
            try:
                v2_context = build_memory_context(message)
                if v2_context:
                    yield {"type": "retrieval_v2", "query": message, "context": v2_context}
                    turn_messages.append({"role": "assistant", "content": v2_context})
            except Exception as v2_exc:
                logger.warning("Memory v2 forced injection failed: %s", v2_exc)

        if runtime_config.get_rag_mode():
            settings = get_settings()
            knowledge_base = get_knowledge_base(self.base_dir)
            try:
                knowledge_results = knowledge_base.search(
                    message,
                    top_k=settings.knowledge_top_k,
                )
            except Exception as exc:
                logger.warning("Knowledge retrieval failed: %s", exc)
                knowledge_results = []
            if knowledge_results:
                yield {
                    "type": "retrieval",
                    "query": message,
                    "results": knowledge_results,
                }
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": knowledge_base.format_context(knowledge_results),
                    }
                )

        turn_messages.append({"role": "user", "content": message})

        agent = self._build_agent()
        run_config: dict[str, Any] = {
            "configurable": {"thread_id": (context.thread_id if context else "")}
        }
        if context and context.callbacks:
            run_config["callbacks"] = context.callbacks
        if not run_config["configurable"]["thread_id"]:
            run_config["configurable"]["thread_id"] = "default"

        final_content_parts: list[str] = []
        last_ai_message = ""
        pending_tools: dict[str, dict[str, str]] = {}
        last_usage: dict[str, Any] | None = None

        async for mode, payload in agent.astream(
            {"messages": turn_messages},
            stream_mode=["messages", "updates"],
            config=run_config,
        ):
            if mode == "messages":
                chunk, metadata = payload
                usage_candidate: Any = None
                if isinstance(metadata, dict):
                    usage_candidate = metadata.get("usage")
                if isinstance(usage_candidate, dict):
                    last_usage = usage_candidate

                node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                if node is not None and node != "agent":
                    continue

                text = _stringify_content(getattr(chunk, "content", ""))
                if text:
                    final_content_parts.append(text)
                    yield {"type": "token", "content": text}
                continue

            if mode != "updates":
                continue

            for update in payload.values():
                if not update:
                    continue
                for agent_message in update.get("messages", []):
                    message_type = getattr(agent_message, "type", "")
                    tool_calls = getattr(agent_message, "tool_calls", []) or []

                    if message_type == "ai" and not tool_calls:
                        candidate = _stringify_content(getattr(agent_message, "content", ""))
                        if candidate:
                            last_ai_message = candidate

                    if tool_calls:
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or tool_call.get("name"))
                            tool_name = str(tool_call.get("name", "tool"))
                            tool_args = tool_call.get("args", "")
                            if not isinstance(tool_args, str):
                                tool_args = json.dumps(tool_args, ensure_ascii=False)
                            pending_tools[call_id] = {
                                "tool": tool_name,
                                "input": str(tool_args),
                            }
                            yield {
                                "type": "tool_start",
                                "tool": tool_name,
                                "input": str(tool_args),
                            }

                    if message_type == "tool":
                        tool_call_id = str(getattr(agent_message, "tool_call_id", ""))
                        pending = pending_tools.pop(
                            tool_call_id,
                            {"tool": getattr(agent_message, "name", "tool"), "input": ""},
                        )
                        output = _stringify_content(getattr(agent_message, "content", ""))
                        yield {
                            "type": "tool_end",
                            "tool": pending["tool"],
                            "output": output,
                        }
                        yield {"type": "new_response"}

        final_content = "".join(final_content_parts).strip() or last_ai_message.strip()
        if last_usage and context and context.callbacks:
            try:
                from langfuse import get_client
                from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

                langfuse_handler: Any | None = None
                for cb in context.callbacks:
                    if isinstance(cb, LangfuseCallbackHandler):
                        langfuse_handler = cb
                        break
                trace_id = getattr(langfuse_handler, "last_trace_id", None) if langfuse_handler else None
                if trace_id:
                    client = get_client()
                    client.trace.update(
                        id=trace_id,
                        usage={
                            "input": last_usage.get("prompt_tokens", 0),
                            "output": last_usage.get("completion_tokens", 0),
                            "total": last_usage.get("total_tokens", 0),
                        },
                    )
            except Exception as exc:
                print("[langfuse] failed to update usage:", repr(exc))
        yield {"type": "done", "content": final_content}

    async def generate_title(self, first_user_message: str) -> str:
        prompt = (
            "Generate a concise Chinese conversation title from the user's first message. "
            "Keep it within 10 Chinese characters when possible. Return only the title."
        )
        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": first_user_message},
                ]
            )
            title = _stringify_content(getattr(response, "content", "")).strip()
            return title[:10] or "New Chat"
        except Exception:
            return (first_user_message.strip() or "New Chat")[:10]

    async def summarize_history(self, messages: list[dict[str, Any]]) -> str:
        prompt = (
            "Summarize the conversation in Chinese within 500 characters. "
            "Preserve user goals, completed steps, key conclusions, and unresolved items."
        )
        lines: list[str] = []
        for item in messages:
            role = item.get("role", "assistant")
            content = str(item.get("content", "") or "")
            if content:
                lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)

        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ]
            )
            summary = _stringify_content(getattr(response, "content", "")).strip()
            return summary[:500]
        except Exception:
            return transcript[:500]


agent_manager = AgentManager()
