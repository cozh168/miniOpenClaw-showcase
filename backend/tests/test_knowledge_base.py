from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

graph_llm_module = sys.modules.get("graph.llm")
if graph_llm_module is None:
    graph_llm_module = types.ModuleType("graph.llm")
    sys.modules["graph.llm"] = graph_llm_module

if not hasattr(graph_llm_module, "build_embedding_config_from_settings"):
    graph_llm_module.build_embedding_config_from_settings = lambda *_args, **_kwargs: {}
if not hasattr(graph_llm_module, "get_embedding_model"):
    graph_llm_module.get_embedding_model = lambda *_args, **_kwargs: None

from service.knowledge_base import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def test_search_markdown_returns_matching_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_dir = root / "knowledge" / "Safety"
            knowledge_dir.mkdir(parents=True)
            (knowledge_dir / "xss.md").write_text(
                "XSS 是跨站脚本攻击，常见防护包括转义输出与输入校验。",
                encoding="utf-8",
            )

            knowledge_base = KnowledgeBase(root)
            with patch.object(knowledge_base, "_supports_dense", return_value=False):
                results = knowledge_base.search("什么是 XSS？", top_k=2)

        self.assertTrue(results)
        self.assertIn("knowledge/Safety/xss.md", results[0]["source"])
        self.assertIn("XSS", results[0]["text"])

    def test_search_json_returns_matching_faq_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_dir = root / "knowledge" / "Store"
            knowledge_dir.mkdir(parents=True)
            (knowledge_dir / "faq.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "未付款订单多久自动取消？",
                            "answer": "未付款订单会在 24 小时后自动取消。",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            knowledge_base = KnowledgeBase(root)
            with patch.object(knowledge_base, "_supports_dense", return_value=False):
                results = knowledge_base.search("未付款订单多久会自动取消？", top_k=1)

        self.assertTrue(results)
        self.assertIn("knowledge/Store/faq.json", results[0]["source"])
        self.assertIn("24", results[0]["text"])


if __name__ == "__main__":
    unittest.main()
