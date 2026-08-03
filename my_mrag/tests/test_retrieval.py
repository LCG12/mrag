from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_mrag.config import Settings
from my_mrag.retrieval import RetrievalConfig, RetrievalPipeline
from my_mrag.schemas import ContentItem, ContentType, ParsedDocument, TextChunk
from my_mrag.storage import JsonDocumentStore, JsonTextChunkStore


class FakeEmbeddingFunc:
    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    async def __call__(self, texts: list[str], context: str):
        self.calls.append((texts, context))
        return [[0.1, 0.2]]


class FakeVectorStorage:
    def __init__(self, hits: list[dict[str, Any]]):
        self.hits = hits
        self.query_embeddings: list[list[float]] = []

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        self.query_embeddings.append(query_embedding)
        return self.hits[:top_k]


class FakeTextChunkStorage:
    def __init__(self, records: dict[str, dict[str, Any]]):
        self.records = records

    async def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        return self.records.get(chunk_id)


class FakeGraphStorage:
    def __init__(self):
        self.nodes = {
            "Architecture (image)": {
                "entity_type": "image",
                "description": "A planner and executor architecture.",
                "source_id": "internal-image",
            },
            "DOCUMENT::doc-1": {
                "entity_type": "document",
                "description": "The source paper.",
                "source_id": "internal-text",
            },
        }
        self.edge = {
            "description": "The architecture belongs to the paper.",
            "keywords": "belongs_to,source_document",
            "weight": 10.0,
            "source_id": "internal-image",
        }

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes.get(node_id)

    async def get_edge(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> dict[str, Any] | None:
        if {source_node_id, target_node_id} == {
            "Architecture (image)",
            "DOCUMENT::doc-1",
        }:
            return self.edge
        return None

    async def get_node_edges(
        self,
        source_node_id: str,
    ) -> list[tuple[str, str]]:
        if source_node_id in self.nodes:
            return [("Architecture (image)", "DOCUMENT::doc-1")]
        return []


class FakeRAG:
    def __init__(self):
        self.embedding_func = FakeEmbeddingFunc()
        self.chunks_vdb = FakeVectorStorage(
            [
                {"id": "internal-other", "distance": 0.99},
                {"id": "internal-text", "distance": 0.9},
            ]
        )
        self.entities_vdb = FakeVectorStorage(
            [
                {
                    "entity_name": "Architecture (image)",
                    "distance": 0.8,
                    "source_id": "internal-image",
                }
            ]
        )
        self.relationships_vdb = FakeVectorStorage([])
        self.text_chunks = FakeTextChunkStorage(
            {
                "internal-other": {
                    "_id": "internal-other",
                    "content": "Evidence from a different document.",
                    "source_id": "other-chunk",
                    "full_doc_id": "doc-2",
                    "chunk_order_index": 0,
                },
                "internal-text": {
                    "_id": "internal-text",
                    "content": "The planner retrieves evidence before execution.",
                    "source_id": "text-chunk-1",
                    "full_doc_id": "doc-1",
                    "chunk_order_index": 1,
                    "file_path": "paper.pdf",
                },
                "internal-image": {
                    "_id": "internal-image",
                    "content": "Image Content Analysis: planner architecture.",
                    "source_id": "item-image",
                    "full_doc_id": "doc-1",
                    "chunk_order_index": 2,
                    "file_path": "paper.pdf",
                },
            }
        )
        self.chunk_entity_relation_graph = FakeGraphStorage()


def _settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    settings = Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        parsed_dir=data_dir / "parsed",
        assets_dir=data_dir / "assets",
        analysis_dir=data_dir / "analysis",
        chunks_dir=data_dir / "chunks",
        knowledge_dir=data_dir / "knowledge",
        lightrag_dir=data_dir / "lightrag",
    )
    settings.ensure_directories()
    return settings


def _store_sources(settings: Settings) -> None:
    document = ParsedDocument(
        document_id="doc-1",
        source_path="paper.pdf",
        parser="test",
        page_count=2,
        items=[
            ContentItem(
                item_id="item-text",
                document_id="doc-1",
                type=ContentType.TEXT,
                page_idx=0,
                order_idx=1,
                text="The planner retrieves evidence before execution.",
            ),
            ContentItem(
                item_id="item-image",
                document_id="doc-1",
                type=ContentType.IMAGE,
                page_idx=1,
                order_idx=2,
                asset_path="figure-1.png",
                captions=["Figure 1: Planner architecture"],
            ),
        ],
    )
    JsonDocumentStore(settings.parsed_dir).save(document)
    JsonTextChunkStore(settings.chunks_dir).save(
        document.document_id,
        [
            TextChunk(
                chunk_id="text-chunk-1",
                document_id="doc-1",
                chunk_index=0,
                text="The planner retrieves evidence before execution.",
                token_count=7,
                page_start=0,
                page_end=0,
                source_order_start=1,
                source_item_ids=("item-text",),
                section_path=("2 METHODS",),
            )
        ],
    )


def test_retrieval_merges_vector_graph_and_source_metadata(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _store_sources(settings)
    rag = FakeRAG()
    pipeline = RetrievalPipeline(
        rag,
        settings,
        RetrievalConfig(
            chunk_top_k=2,
            entity_top_k=1,
            relationship_top_k=0,
            final_top_k=5,
            candidate_multiplier=1,
            graph_depth=1,
        ),
    )

    result = asyncio.run(
        pipeline.retrieve(
            "How does the planner work?",
            document_id="doc-1",
        )
    )

    assert len(rag.embedding_func.calls) == 1
    assert all(
        storage.query_embeddings == [[0.1, 0.2]]
        for storage in (
            rag.chunks_vdb,
            rag.entities_vdb,
        )
    )
    assert [chunk.source_id for chunk in result.chunks] == [
        "text-chunk-1",
        "item-image",
    ]
    text_hit, image_hit = result.chunks
    assert text_hit.page_start == 1
    assert text_hit.section_path == ("2 METHODS",)
    assert text_hit.channels == ("chunk_vector",)
    assert image_hit.content_type == "image"
    assert image_hit.page_start == 2
    assert image_hit.asset_path == "figure-1.png"
    assert image_hit.captions == ("Figure 1: Planner architecture",)
    assert "entity_source" in image_hit.channels
    assert {entity.entity_name for entity in result.entities} == {
        "Architecture (image)",
        "DOCUMENT::doc-1",
    }
    assert len(result.relationships) == 1
    assert result.relationships[0].channels == ("graph_expand",)


def test_retrieval_rejects_empty_query(tmp_path: Path) -> None:
    pipeline = RetrievalPipeline(FakeRAG(), _settings(tmp_path))

    try:
        asyncio.run(pipeline.retrieve("   "))
    except ValueError as error:
        assert "cannot be empty" in str(error)
    else:
        raise AssertionError("Expected an empty query to be rejected")
