from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from my_mrag.indexing import LightRAGIndexer
from my_mrag.schemas import (
    ChunkKnowledge,
    ContentItem,
    ContentType,
    EntityInfo,
    KnowledgeEntity,
    KnowledgeRelationship,
    ModalAnalysis,
    ParsedDocument,
    TextChunk,
)


class FakeCustomKGStore:
    def __init__(self):
        self.calls: list[tuple[dict[str, object], str | None]] = []

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, object],
        full_doc_id: str | None = None,
    ) -> None:
        self.calls.append((custom_kg, full_doc_id))


def _document() -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_path=str(Path("papers") / "reason-plan-react.pdf"),
        parser="test",
        page_count=1,
        items=[
            ContentItem(
                item_id="item-image",
                document_id="doc-1",
                type=ContentType.IMAGE,
                page_idx=0,
                order_idx=4,
                asset_path="figure.png",
            )
        ],
        metadata={"title": "Reason-Plan-ReAct"},
    )


def _analysis() -> ModalAnalysis:
    return ModalAnalysis(
        item_id="item-image",
        document_id="doc-1",
        content_type=ContentType.IMAGE,
        detailed_description="The diagram separates planning and execution.",
        entity_info=EntityInfo(
            entity_name="RP-ReAct architecture",
            entity_type="image",
            summary="A planner delegates tool execution to proxy agents.",
        ),
        context="Surrounding paper text.",
        chunk_text=(
            "Image Content Analysis:\n"
            "Visual Analysis: The planner delegates work to proxy agents."
        ),
        model_name="fake-vision",
    )


def test_indexer_builds_chunks_entities_and_document_relations() -> None:
    store = FakeCustomKGStore()
    indexer = LightRAGIndexer(store)

    report = asyncio.run(indexer.index(_document(), [_analysis()]))

    assert report.chunk_count == 1
    assert report.entity_count == 2
    assert report.relationship_count == 1
    payload, full_doc_id = store.calls[0]
    assert full_doc_id == "doc-1"

    chunk = payload["chunks"][0]
    assert chunk["source_id"] == "item-image"
    assert chunk["chunk_order_index"] == 4
    assert "Visual Analysis" in chunk["content"]

    modal_entity, document_entity = payload["entities"]
    assert modal_entity["entity_name"] == "RP-ReAct architecture (image)"
    assert modal_entity["source_id"] == "item-image"
    assert document_entity["entity_name"] == "DOCUMENT::doc-1"

    relationship = payload["relationships"][0]
    assert relationship["src_id"] == "RP-ReAct architecture (image)"
    assert relationship["tgt_id"] == "DOCUMENT::doc-1"
    assert "belongs_to" in relationship["keywords"]
    assert relationship["source_id"] == "item-image"


def test_indexer_rejects_duplicate_analyses() -> None:
    indexer = LightRAGIndexer(FakeCustomKGStore())
    with pytest.raises(ValueError, match="Duplicate analysis"):
        indexer.build_payload(_document(), [_analysis(), _analysis()])


def test_indexer_accepts_text_chunks_without_multimodal_analysis() -> None:
    document = _document()
    document.items.append(
        ContentItem(
            item_id="item-text",
            document_id="doc-1",
            type=ContentType.TEXT,
            page_idx=0,
            order_idx=1,
            text="The planner retrieves evidence before execution.",
        )
    )
    text_chunk = TextChunk(
        chunk_id="chunk-text-1",
        document_id="doc-1",
        chunk_index=0,
        text="The planner retrieves evidence before execution.",
        token_count=8,
        page_start=0,
        page_end=0,
        source_order_start=1,
        source_item_ids=("item-text",),
        section_path=("2 METHODS",),
    )
    store = FakeCustomKGStore()

    report = asyncio.run(
        LightRAGIndexer(store).index(document, [], [text_chunk])
    )

    assert report.chunk_count == 1
    assert report.text_chunk_count == 1
    assert report.multimodal_chunk_count == 0
    payload, full_doc_id = store.calls[0]
    assert full_doc_id == "doc-1"
    assert payload["chunks"][0]["source_id"] == "chunk-text-1"
    assert payload["chunks"][0]["content"].startswith(
        "Section: 2 METHODS"
    )
    assert payload["entities"][0]["entity_name"] == "DOCUMENT::doc-1"


def test_indexer_merges_text_entities_and_relationships() -> None:
    document = _document()
    document.items.append(
        ContentItem(
            item_id="item-text",
            document_id="doc-1",
            type=ContentType.TEXT,
            page_idx=0,
            order_idx=1,
            text="RAG-Anything constructs a dual graph.",
        )
    )
    text_chunk = TextChunk(
        chunk_id="chunk-text-1",
        document_id="doc-1",
        chunk_index=0,
        text="RAG-Anything constructs a dual graph.",
        token_count=7,
        page_start=0,
        page_end=0,
        source_order_start=1,
        source_item_ids=("item-text",),
        section_path=("2 METHODS",),
    )
    knowledge = ChunkKnowledge(
        chunk_id=text_chunk.chunk_id,
        document_id="doc-1",
        chunk_index=0,
        entities=(
            KnowledgeEntity(
                entity_name="RAG-Anything",
                entity_type="method",
                description="A multimodal retrieval framework.",
            ),
            KnowledgeEntity(
                entity_name="RAG Anything",
                entity_type="method",
                description="The same framework without punctuation.",
            ),
            KnowledgeEntity(
                entity_name="Dual-Graph Construction",
                entity_type="component",
                description="Builds cross-modal and modality-aware graphs.",
            ),
        ),
        relationships=(
            KnowledgeRelationship(
                source_entity="RAG Anything",
                target_entity="Dual-Graph Construction",
                description="The framework uses dual-graph construction.",
                keywords=("uses", "constructs"),
                weight=9,
            ),
        ),
        model_name="fake-text-model",
    )
    store = FakeCustomKGStore()

    report = asyncio.run(
        LightRAGIndexer(store).index(
            document,
            [],
            [text_chunk],
            [knowledge],
        )
    )

    assert report.knowledge_chunk_count == 1
    assert report.text_entity_count == 3
    assert report.text_relationship_count == 1
    payload, _ = store.calls[0]
    entities = {
        entity["entity_name"]: entity for entity in payload["entities"]
    }
    assert entities["RAG-Anything"]["source_id"] == "chunk-text-1"
    assert "without punctuation" in entities["RAG-Anything"]["description"]
    assert entities["Dual-Graph Construction"]["entity_type"] == "component"
    relationship = payload["relationships"][0]
    assert relationship["src_id"] == "RAG-Anything"
    assert relationship["tgt_id"] == "Dual-Graph Construction"
    assert relationship["source_id"] == "chunk-text-1"
    assert relationship["keywords"] == "uses,constructs"


def test_real_lightrag_storages_receive_custom_kg(tmp_path: Path) -> None:
    lightrag = pytest.importorskip("lightrag")
    numpy = pytest.importorskip("numpy")
    from lightrag.utils import EmbeddingFunc, compute_mdhash_id

    async def embed(texts: list[str]):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = numpy.asarray(
                [byte + 1 for byte in digest[:8]],
                dtype=numpy.float32,
            )
            vector /= numpy.linalg.norm(vector)
            vectors.append(vector)
        return numpy.stack(vectors)

    async def unused_llm(*args, **kwargs) -> str:
        raise AssertionError("Custom KG indexing must not call the LLM")

    async def run() -> None:
        rag = lightrag.LightRAG(
            working_dir=str(tmp_path / "lightrag"),
            llm_model_func=unused_llm,
            llm_model_name="test-only",
            embedding_func=EmbeddingFunc(
                embedding_dim=8,
                max_token_size=8192,
                func=embed,
            ),
        )
        await rag.initialize_storages()
        try:
            analysis = _analysis()
            await LightRAGIndexer(rag).index(_document(), [analysis])

            chunk_id = compute_mdhash_id(
                analysis.chunk_text,
                prefix="chunk-",
            )
            stored_chunk = await rag.text_chunks.get_by_id(chunk_id)
            assert stored_chunk is not None
            assert stored_chunk["full_doc_id"] == "doc-1"

            entity_name = "RP-ReAct architecture (image)"
            document_entity = "DOCUMENT::doc-1"
            node = await rag.chunk_entity_relation_graph.get_node(entity_name)
            assert node is not None
            assert node["entity_type"] == "image"
            assert await rag.chunk_entity_relation_graph.has_edge(
                entity_name,
                document_entity,
            )

            chunk_hits = await rag.chunks_vdb.query(
                analysis.chunk_text,
                top_k=1,
            )
            entity_hits = await rag.entities_vdb.query(
                "RP-ReAct architecture",
                top_k=1,
            )
            relation_hits = await rag.relationships_vdb.query(
                "source document",
                top_k=1,
            )
            assert chunk_hits
            assert entity_hits
            assert relation_hits
        finally:
            await rag.finalize_storages()

    asyncio.run(run())
