from __future__ import annotations

import asyncio

import pytest

from my_mrag.answering import AnswerPipeline
from my_mrag.schemas import (
    AnalysisRequest,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
)


class FakeRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result
        self.calls: list[tuple[str, str | None]] = []

    async def retrieve(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> RetrievalResult:
        self.calls.append((query, document_id))
        return self.result


class FakeModel:
    model_name = "test-answer-model"

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.requests: list[AnalysisRequest] = []

    async def complete(self, request: AnalysisRequest) -> str:
        self.requests.append(request)
        return self.responses.pop(0)


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="How does RP-ReAct work?",
        chunks=(
            RetrievedChunk(
                chunk_id="chunk-text",
                source_id="source-text",
                document_id="doc-paper",
                file_path="papers/rp-react.pdf",
                page_start=2,
                page_end=2,
                section_path=("3 Method", "3.1 Planner"),
                content_type="text",
                content=(
                    "The reasoner planner creates a high-level plan and "
                    "delegates tool execution to proxy agents."
                ),
                score=0.91,
            ),
            RetrievedChunk(
                chunk_id="chunk-equation",
                source_id="source-equation",
                document_id="doc-paper",
                file_path="papers/rp-react.pdf",
                page_start=7,
                page_end=7,
                section_path=("5 Experiments",),
                content_type="equation",
                content="CPS combines normalized accuracy and saturation.",
                score=0.82,
            ),
        ),
        entities=(
            RetrievedEntity(
                entity_name="Reasoner Planner",
                entity_type="component",
                description="Creates high-level plans.",
                score=0.88,
            ),
        ),
        relationships=(
            RetrievedRelationship(
                source_entity="Reasoner Planner",
                target_entity="Proxy Agent",
                description="Delegates low-level execution.",
                keywords="delegates_to",
                score=0.84,
            ),
        ),
    )


def test_answer_builds_grounded_context_and_source_map() -> None:
    retriever = FakeRetriever(_retrieval_result())
    model = FakeModel(
        [
            "The planner delegates low-level tool execution [S1]. "
            "CPS combines accuracy and saturation [S2]."
        ]
    )

    result = asyncio.run(
        AnswerPipeline(retriever, model).answer(
            "How does RP-ReAct work?",
            document_id="doc-paper",
        )
    )

    assert retriever.calls == [("How does RP-ReAct work?", "doc-paper")]
    assert len(model.requests) == 1
    request = model.requests[0]
    assert "Question:\nHow does RP-ReAct work?" in request.prompt
    assert "[S1]" in request.prompt
    assert "Document: rp-react.pdf" in request.prompt
    assert "PDF pages: 2" in request.prompt
    assert "Section: 3 Method > 3.1 Planner" in request.prompt
    assert "Content type: equation" in request.prompt
    assert "[E1] Reasoner Planner" in request.prompt
    assert "[R1] Reasoner Planner -> Proxy Agent" in request.prompt
    assert result.cited_source_ids == ("S1", "S2")
    assert [source.chunk_id for source in result.sources] == [
        "chunk-text",
        "chunk-equation",
    ]
    assert result.sources[0].page_start == 2
    assert result.sources[1].content_type == "equation"
    assert result.retrieved_entity_count == 1
    assert result.retrieved_relationship_count == 1
    assert result.to_dict()["evidence_counts"] == {
        "sources": 2,
        "entities": 1,
        "relationships": 1,
    }


@pytest.mark.parametrize(
    ("invalid_answer", "expected_error"),
    [
        ("The planner delegates execution [S99].", "unknown sources"),
        (
            "The planner delegates execution [S1][E1].",
            "graph hints",
        ),
    ],
)
def test_answer_retries_invalid_citation(
    invalid_answer: str,
    expected_error: str,
) -> None:
    retriever = FakeRetriever(_retrieval_result())
    model = FakeModel(
        [
            invalid_answer,
            "The planner delegates execution [S1].",
        ]
    )

    result = asyncio.run(
        AnswerPipeline(retriever, model).answer("Explain the planner.")
    )

    assert len(model.requests) == 2
    assert "failed citation validation" in model.requests[1].prompt
    assert expected_error in model.requests[1].prompt
    assert result.cited_source_ids == ("S1",)


def test_answer_rejects_empty_query() -> None:
    retriever = FakeRetriever(_retrieval_result())
    model = FakeModel(["Unused"])

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(AnswerPipeline(retriever, model).answer("   "))

    assert retriever.calls == []
    assert model.requests == []
