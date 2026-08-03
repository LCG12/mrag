from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from my_mrag.knowledge import (
    KnowledgeExtractionConfig,
    KnowledgeExtractionPipeline,
)
from my_mrag.schemas import AnalysisRequest, TextChunk
from my_mrag.storage import JsonKnowledgeStore


class FakeKnowledgeModel:
    model_name = "fake-knowledge-model"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.requests: list[AnalysisRequest] = []

    async def complete(self, request: AnalysisRequest) -> str:
        self.requests.append(request)
        return self.responses.pop(0)


def _chunk(chunk_id: str = "chunk-1", chunk_index: int = 0) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=chunk_index,
        text=(
            "RAG-Anything constructs a cross-modal knowledge graph and a "
            "modality-aware graph."
        ),
        token_count=14,
        page_start=3,
        page_end=4,
        source_order_start=8,
        source_item_ids=("item-1",),
        section_path=("2 FRAMEWORK", "2.2 DUAL-GRAPH CONSTRUCTION"),
    )


def _valid_response() -> str:
    return """<think>private reasoning</think>
```json
{
  "entities": [
    {
      "entity_name": "RAG-Anything",
      "entity_type": "method",
      "description": "A multimodal retrieval framework."
    },
    {
      "entity_name": "Cross-Modal Knowledge Graph",
      "entity_type": "component",
      "description": "Grounds non-textual content in textual context."
    }
  ],
  "relationships": [
    {
      "source_entity": "rag-anything",
      "target_entity": "Cross-Modal Knowledge Graph",
      "description": "RAG-Anything constructs the graph.",
      "keywords": ["constructs", "contains"],
      "weight": 8
    }
  ]
}
```"""


def test_pipeline_prepares_parses_and_canonicalizes_knowledge() -> None:
    model = FakeKnowledgeModel([_valid_response()])
    pipeline = KnowledgeExtractionPipeline(model)

    result = asyncio.run(pipeline.extract_chunk(_chunk()))

    assert len(model.requests) == 1
    request = model.requests[0]
    assert "2 FRAMEWORK > 2.2 DUAL-GRAPH CONSTRUCTION" in request.prompt
    assert "Pages: 4-5" in request.prompt
    assert "<source_text>" in request.prompt
    assert [entity.entity_name for entity in result.entities] == [
        "RAG-Anything",
        "Cross-Modal Knowledge Graph",
    ]
    relationship = result.relationships[0]
    assert relationship.source_entity == "RAG-Anything"
    assert relationship.target_entity == "Cross-Modal Knowledge Graph"
    assert relationship.keywords == ("constructs", "contains")
    assert relationship.weight == 8
    assert result.model_name == "fake-knowledge-model"


def test_pipeline_retries_invalid_relationship_endpoint() -> None:
    invalid = _valid_response().replace(
        '"target_entity": "Cross-Modal Knowledge Graph"',
        '"target_entity": "Missing Entity"',
    )
    model = FakeKnowledgeModel([invalid, _valid_response()])
    pipeline = KnowledgeExtractionPipeline(
        model,
        KnowledgeExtractionConfig(retries=1),
    )

    result = asyncio.run(pipeline.extract_chunk(_chunk()))

    assert len(model.requests) == 2
    assert len(result.relationships) == 1


def test_pipeline_rejects_non_json_after_retries() -> None:
    model = FakeKnowledgeModel(["not json"])
    pipeline = KnowledgeExtractionPipeline(
        model,
        KnowledgeExtractionConfig(retries=0),
    )

    with pytest.raises(RuntimeError, match="chunk-1"):
        asyncio.run(pipeline.extract_chunk(_chunk()))


def test_knowledge_store_merges_and_round_trips(tmp_path: Path) -> None:
    model = FakeKnowledgeModel([_valid_response(), _valid_response()])
    pipeline = KnowledgeExtractionPipeline(model)
    first = asyncio.run(pipeline.extract_chunk(_chunk()))
    second = asyncio.run(
        pipeline.extract_chunk(_chunk("chunk-2", chunk_index=1))
    )
    store = JsonKnowledgeStore(tmp_path / "knowledge")

    store.save("doc-1", [first])
    merged, stored_path = store.merge_and_save("doc-1", [second])

    assert stored_path.is_file()
    assert store.load("doc-1") == merged == [first, second]
    payload = json.loads(stored_path.read_text(encoding="utf-8"))
    assert payload["chunk_count"] == 2
    assert payload["entity_count"] == 4
    assert payload["relationship_count"] == 2
