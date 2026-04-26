from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.tool_security import router as tool_security_router
from config import Settings
from graph.agent import agent_manager
from service.tool_security import ToolSecurityManager
from tools.fetch_url_tool import FetchURLTool
from tools.read_file_tool import ReadFileTool
from tools.terminal_tool import TerminalTool


def make_settings(root: Path, **overrides) -> Settings:
    payload = {
        "config_dir": root / "config",
        "backend_dir": root,
        "project_root": root.parent,
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "llm_api_key": None,
        "llm_base_url": "https://api.openai.com/v1",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_key": None,
        "embedding_base_url": "https://api.openai.com/v1",
    }
    payload.update(overrides)
    return Settings(**payload)


class ToolSecurityTests(unittest.TestCase):
    def test_terminal_allowlist_blocks_disallowed_command_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            security = ToolSecurityManager(
                root,
                settings=make_settings(root, terminal_allowed_commands=("python",)),
            )
            tool = TerminalTool(root_dir=root, security_manager=security)

            result = tool._run("Get-ChildItem")

            self.assertIn("Blocked:", result)
            entries = security.audit.list_entries(limit=1)
            self.assertEqual("terminal", entries[0]["tool"])
            self.assertEqual("blocked", entries[0]["status"])

    def test_fetch_url_blocks_localhost_before_network_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            security = ToolSecurityManager(root, settings=make_settings(root))
            tool = FetchURLTool(root_dir=root, security_manager=security)

            result = tool._run("http://127.0.0.1:8002/health")

            self.assertIn("Fetch failed:", result)
            entries = security.audit.list_entries(limit=1)
            self.assertEqual("fetch_url", entries[0]["tool"])
            self.assertEqual("blocked", entries[0]["status"])

    def test_read_file_blocks_sensitive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "backend" / "config"
            target.mkdir(parents=True)
            (target / ".env").write_text("SECRET=1", encoding="utf-8")

            security = ToolSecurityManager(root, settings=make_settings(root))
            tool = ReadFileTool(root_dir=root, security_manager=security)

            result = tool._run("backend/config/.env")

            self.assertIn("Read failed:", result)
            entries = security.audit.list_entries(limit=1)
            self.assertEqual("read_file", entries[0]["tool"])
            self.assertEqual("blocked", entries[0]["status"])

    def test_tool_security_api_exposes_policy_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            security = ToolSecurityManager(
                root,
                settings=make_settings(root, terminal_allowed_commands=("python",)),
            )
            security.audit.record(
                tool="terminal",
                status="blocked",
                input_value="Get-ChildItem",
                reason="command not allowed",
            )

            app = FastAPI()
            app.include_router(tool_security_router, prefix="/api")

            with patch("api.tool_security.get_tool_security_manager", return_value=security):
                with patch.object(agent_manager, "base_dir", root):
                    client = TestClient(app)
                    policy_response = client.get("/api/tool-security")
                    audit_response = client.get("/api/tool-audit?limit=5")

            self.assertEqual(200, policy_response.status_code)
            self.assertEqual(["python"], policy_response.json()["policy"]["terminal_allowed_commands"])
            self.assertEqual(200, audit_response.status_code)
            self.assertEqual("terminal", audit_response.json()["items"][0]["tool"])


if __name__ == "__main__":
    unittest.main()
