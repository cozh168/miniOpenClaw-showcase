from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import get_settings
from graph.llm import build_embedding_config_from_settings, get_embedding_model

CHROMA_MEMORY_COLLECTION = "memory"
CHROMA_DEFAULT_PERSIST_DIR = "storage/chroma_memory"


class MemoryIndexer:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._vector_store: Any = None
        self._embedding: Embeddings | None = None

    def configure(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _memory_path(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("MemoryIndexer is not configured")
        return self.base_dir / "memory_module_v1" / "long_term_memory" / "MEMORY.md"

    @property
    def _storage_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError("MemoryIndexer is not configured")
        persist = os.getenv("CHROMA_PERSIST_DIR", "").strip()
        if persist:
            return Path(persist)
        return self.base_dir / CHROMA_DEFAULT_PERSIST_DIR.replace("/", os.sep)

    @property
    def _meta_path(self) -> Path:
        return self._storage_dir / "meta.json"

    def _supports_embeddings(self) -> bool:
        return bool(get_settings().embedding_api_key)

    def _get_embedding(self) -> Embeddings:
        if self._embedding is None:
            settings = get_settings()
            config = build_embedding_config_from_settings(settings)
            self._embedding = get_embedding_model(config)
        return self._embedding

    def _file_digest(self) -> str:
        if not self._memory_path.exists():
            return ""
        return hashlib.md5(self._memory_path.read_bytes()).hexdigest()

    def _read_meta(self) -> dict[str, Any]:
        if not self._meta_path.exists():
            return {}
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_meta(self, digest: str) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps({"digest": digest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_documents(self) -> list[Document]:
        content = self._memory_path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        chunk_size, overlap = 256, 32
        chunks: list[str] = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(content[start:end])
            start = end - overlap if end < len(content) else len(content)

        return [
            Document(
                page_content=chunk,
                metadata={"source": "memory_module_v1/long_term_memory/MEMORY.md"},
            )
            for chunk in chunks
            if chunk.strip()
        ]

    def rebuild_index(self) -> None:
        if self.base_dir is None:
            return

        if not self._memory_path.exists():
            self._memory_path.write_text("# Long-term Memory\n\n", encoding="utf-8")

        digest = self._file_digest()
        self._vector_store = None

        if not self._supports_embeddings():
            self._write_meta(digest)
            return

        try:
            from langchain_chroma import Chroma

            documents = self._build_documents()
            if self._storage_dir.exists():
                shutil.rmtree(self._storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)

            if documents:
                self._vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding_function=self._get_embedding(),
                    persist_directory=str(self._storage_dir),
                    collection_name=CHROMA_MEMORY_COLLECTION,
                )
            self._write_meta(digest)
        except Exception as exc:
            self._vector_store = None
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            logger_payload = {"digest": digest, "last_error": str(exc)}
            self._meta_path.write_text(
                json.dumps(logger_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_index(self) -> None:
        if not self._supports_embeddings():
            self._vector_store = None
            return
        try:
            from langchain_chroma import Chroma

            self._vector_store = Chroma(
                persist_directory=str(self._storage_dir),
                embedding_function=self._get_embedding(),
                collection_name=CHROMA_MEMORY_COLLECTION,
            )
        except Exception:
            self._vector_store = None

    def _maybe_rebuild(self) -> None:
        if self.base_dir is None:
            return
        digest = self._file_digest()
        if digest != self._read_meta().get("digest"):
            self.rebuild_index()
            return
        if self._vector_store is None and self._supports_embeddings():
            self._load_index()

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.base_dir is None:
            return []

        self._maybe_rebuild()
        if self._vector_store is None:
            return []

        try:
            docs_with_scores = self._vector_store.similarity_search_with_score(query, k=top_k)
        except Exception:
            return []

        payload: list[dict[str, Any]] = []
        for doc, score in docs_with_scores:
            payload.append(
                {
                    "text": doc.page_content,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                    "source": doc.metadata.get(
                        "source",
                        "memory_module_v1/long_term_memory/MEMORY.md",
                    ),
                }
            )
        return payload


memory_indexer = MemoryIndexer()
