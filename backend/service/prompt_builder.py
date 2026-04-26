from __future__ import annotations

from pathlib import Path

from config import get_settings, runtime_config
from memory_module_v2.service.config import get_memory_backend, get_memory_v2_inject_mode

SYSTEM_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Skills Snapshot", "skills/SKILLS_SNAPSHOT.md"),
    ("Soul", "workspace/SOUL.md"),
    ("Identity", "workspace/IDENTITY.md"),
    ("User Profile", "workspace/USER.md"),
    ("Agents Guide", "workspace/AGENTS.md"),
)

_MEMORY_V1_PATH = "memory_module_v1/long_term_memory/MEMORY.md"

_MEMORY_HINTS = {
    "off": (
        "<!-- Long-term Memory -->\n"
        "Long-term memory is disabled for this run. Do not infer private user facts "
        "from old memory files or previous demo sessions unless they appear in the "
        "current conversation."
    ),
    "v1_rag": (
        "<!-- Long-term Memory -->\n"
        "Long-term memory is retrieved dynamically from MEMORY.md. Use only memory "
        "snippets that are retrieved for this turn; do not assume unretrieved memory "
        "is still valid."
    ),
    "v2_tool": (
        "<!-- Long-term Memory (v2) -->\n"
        "You can call the `search_memory` tool to retrieve long-term memory across "
        "sessions. Use it when the user asks about past discussions or prior context. "
        "Prefer verbatim evidence returned by the tool."
    ),
    "v2_always": (
        "<!-- Long-term Memory (v2) -->\n"
        "The system injects relevant long-term memory before each turn. Prefer the "
        "injected evidence when answering history-related questions, and do not assume "
        "un-injected memory is valid."
    ),
}

_KNOWLEDGE_RAG_HINT = (
    "<!-- Document RAG -->\n"
    "When document RAG is enabled, the system may inject `[Knowledge base context]` "
    "from files under the local `knowledge/` directory. For document-related questions, "
    "prefer that context and cite the returned `Source` path when useful. If the injected "
    "context is insufficient, call `search_knowledge_base` to retrieve more."
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _read_component(base_dir: Path, relative_path: str, limit: int) -> str:
    path = base_dir / relative_path
    if not path.exists():
        return f"[missing component: {relative_path}]"
    return _truncate(path.read_text(encoding="utf-8"), limit)


def _get_memory_hint_key() -> str:
    backend = get_memory_backend()
    if backend == "v1":
        return "v1_rag"
    if backend == "v2":
        mode = get_memory_v2_inject_mode()
        return "v2_tool" if mode == "tool" else "v2_always"
    return "off"


def build_system_prompt(base_dir: Path) -> str:
    settings = get_settings()
    parts: list[str] = [
        "<!-- Runtime Model -->\n"
        f"Current chat model provider: {settings.llm_provider}\n"
        f"Current chat model name: {settings.llm_model}\n"
        "When the user asks what model you are using, answer from this runtime block. "
        "This block overrides older conversation text, demo sessions, memory files, "
        "and project names. Do not claim to be Claude, Anthropic, or any other provider "
        "unless it is shown here."
    ]

    for label, relative_path in SYSTEM_COMPONENTS:
        content = _read_component(base_dir, relative_path, settings.component_char_limit)
        parts.append(f"<!-- {label} -->\n{content}")

    hint_key = _get_memory_hint_key()
    if hint_key == "v1_rag":
        parts.append(_MEMORY_HINTS["v1_rag"])
    elif hint_key == "off":
        parts.append(_MEMORY_HINTS["off"])

    memory_hint = _MEMORY_HINTS.get(hint_key)
    if memory_hint and hint_key.startswith("v2_"):
        parts.append(memory_hint)

    if runtime_config.get_rag_mode():
        parts.append(_KNOWLEDGE_RAG_HINT)

    return "\n\n".join(parts)
