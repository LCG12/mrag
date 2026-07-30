from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from my_mrag.config import Settings
from my_mrag.embeddings import (
    EmbeddingSettings,
    build_embedding_model,
    load_embedding_settings,
)


async def _indexing_only_llm(*args: Any, **kwargs: Any) -> str:
    raise RuntimeError(
        "The indexing-only LightRAG runtime cannot call an LLM"
    )


def build_lightrag(
    settings: Settings,
    embedding_settings: EmbeddingSettings,
) -> Any:
    try:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
    except ImportError as exc:
        raise RuntimeError(
            "Install LightRAG with: pip install -e .[graph]"
        ) from exc

    settings.lightrag_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = build_embedding_model(embedding_settings)
    return LightRAG(
        working_dir=str(settings.lightrag_dir),
        llm_model_func=_indexing_only_llm,
        llm_model_name="indexing-only",
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_settings.dimension,
            max_token_size=embedding_settings.max_token_size,
            model_name=embedding_settings.model_name,
            func=embedding_model,
            supports_asymmetric=True,
        ),
    )


@asynccontextmanager
async def open_lightrag(
    settings: Settings,
    embedding_settings: EmbeddingSettings | None = None,
) -> AsyncIterator[Any]:
    rag = build_lightrag(
        settings,
        embedding_settings or load_embedding_settings(settings),
    )
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        await rag.finalize_storages()
