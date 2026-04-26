from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from service.prompt_builder import build_system_prompt


def _write_workspace(root: Path) -> None:
    for relative_path, content in {
        "skills/SKILLS_SNAPSHOT.md": "<skills></skills>",
        "workspace/SOUL.md": "# Soul\nBe reliable.",
        "workspace/IDENTITY.md": "# Identity\nMini OpenClaw",
        "workspace/USER.md": "# User Profile\nNo personal facts have been confirmed yet.",
        "workspace/AGENTS.md": "# Agents Guide\nUse tools carefully.",
        "memory_module_v1/long_term_memory/MEMORY.md": "# Long-term Memory\nName: Jimmy",
    }.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class PromptBuilderTests(unittest.TestCase):
    def test_runtime_model_identity_is_injected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_workspace(root)

            settings = SimpleNamespace(
                component_char_limit=20_000,
                llm_provider="deepseek",
                llm_model="deepseek-chat",
            )

            with patch("service.prompt_builder.get_settings", return_value=settings):
                with patch("service.prompt_builder.get_memory_backend", return_value="off"):
                    with patch("service.prompt_builder.runtime_config.get_rag_mode", return_value=False):
                        prompt = build_system_prompt(root)

        self.assertIn("Current chat model provider: deepseek", prompt)
        self.assertIn("Current chat model name: deepseek-chat", prompt)
        self.assertIn("Do not claim to be Claude", prompt)

    def test_memory_backend_off_does_not_inline_memory_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_workspace(root)

            settings = SimpleNamespace(
                component_char_limit=20_000,
                llm_provider="deepseek",
                llm_model="deepseek-chat",
            )

            with patch("service.prompt_builder.get_settings", return_value=settings):
                with patch("service.prompt_builder.get_memory_backend", return_value="off"):
                    with patch("service.prompt_builder.runtime_config.get_rag_mode", return_value=False):
                        prompt = build_system_prompt(root)

        self.assertIn("Long-term memory is disabled", prompt)
        self.assertNotIn("Name: Jimmy", prompt)


if __name__ == "__main__":
    unittest.main()
