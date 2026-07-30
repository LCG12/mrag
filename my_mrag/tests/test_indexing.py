from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from my_mrag.indexing import LightRAGIndexer
from my_mrag.schemas import (
    ContentItem,
    ContentType,
    EntityInfo,
    ModalAnalysis,
    ParsedDocument,
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
