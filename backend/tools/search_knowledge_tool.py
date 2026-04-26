from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from service.knowledge_base import get_knowledge_base


class SearchKnowledgeInput(BaseModel):
    query: str = Field(..., description="Question to search in the local knowledge base.")
    top_k: int = Field(default=3, ge=1, le=8, description="Number of chunks to retrieve.")


class SearchKnowledgeBaseTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "Search the local knowledge base under knowledge/. "
        "Use this when the user asks about documents, reports, FAQs, PDFs, or local domain knowledge."
    )
    args_schema: Type[BaseModel] = SearchKnowledgeInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _root_dir: Path = PrivateAttr()

    def __init__(self, root_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_dir = root_dir

    def _run(
        self,
        query: str,
        top_k: int = 3,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        knowledge_base = get_knowledge_base(self._root_dir)
        results = knowledge_base.search(query, top_k=top_k)
        if not results:
            return "No relevant knowledge documents found."
        return knowledge_base.format_context(results)

    async def _arun(
        self,
        query: str,
        top_k: int = 3,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(self._run, query, top_k, None)
