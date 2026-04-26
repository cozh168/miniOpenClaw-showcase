from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from config import runtime_config
from service.tool_security import get_tool_security_manager
from tools.fetch_url_tool import FetchURLTool
from tools.python_repl_tool import PythonReplTool
from tools.read_file_tool import ReadFileTool
from tools.search_knowledge_tool import SearchKnowledgeBaseTool
from tools.terminal_tool import TerminalTool


def get_all_tools(base_dir: Path) -> list[BaseTool]:
    security = get_tool_security_manager(base_dir)
    tools: list[BaseTool] = []

    if security.is_tool_enabled("terminal"):
        tools.append(TerminalTool(root_dir=base_dir, security_manager=security))
    if security.is_tool_enabled("python_repl"):
        tools.append(PythonReplTool(root_dir=base_dir, security_manager=security))
    if security.is_tool_enabled("fetch_url"):
        tools.append(FetchURLTool(root_dir=base_dir, security_manager=security))
    if security.is_tool_enabled("read_file"):
        tools.append(ReadFileTool(root_dir=base_dir, security_manager=security))

    if runtime_config.get_rag_mode():
        tools.append(SearchKnowledgeBaseTool(root_dir=base_dir))

    from memory_module_v2.service.config import get_memory_backend, get_memory_v2_inject_mode
    if get_memory_backend() == "v2" and get_memory_v2_inject_mode() == "tool":
        from memory_module_v2.integrations.tools import get_memory_tools
        tools.extend(get_memory_tools())

    return tools
