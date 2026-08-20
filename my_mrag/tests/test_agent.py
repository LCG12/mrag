from __future__ import annotations

import asyncio
from pathlib import Path

from my_mrag.agent import (
    AgentConfig,
    ExecutorConfig,
    PlannerConfig,
    ProxyExecutor,
    ResearchPlanner,
    RPReActAgent,
)
from my_mrag.answering import AnswerPipeline
from my_mrag.memory import ConversationMemory
from my_mrag.schemas import (
    AnswerResult,
    AnswerSource,
    AnalysisRequest,
    ResearchPlan,
    ResearchPlanStep,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
)
from my_mrag.storage import JsonConversationStore


class FakeModel:
    def __init__(self, model_name: str, responses: list[str]):
        self.model_name = model_name
        self.responses = responses
        self.requests: list[AnalysisRequest] = []

    async def complete(self, request: AnalysisRequest) -> str:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, results: dict[str, RetrievalResult]):
        self.results = results
        self.calls: list[tuple[str, str | None]] = []

    async def retrieve(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> RetrievalResult:
        self.calls.append((query, document_id))
        return self.results[query]


def _chunk(
    chunk_id: str,
    content: str,
    score: float,
    channel: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        document_id="doc-paper",
        file_path="paper.pdf",
        page_start=2,
        page_end=2,
        content=content,
        score=score,
        channels=(channel,),
    )


def _results() -> dict[str, RetrievalResult]:
    shared_entity = RetrievedEntity(
        entity_name="RP-ReAct",
        entity_type="method",
        description="A planner and proxy-executor architecture.",
        score=0.9,
        source_chunk_ids=("common",),
        channels=("entity_vector",),
    )
    shared_relationship = RetrievedRelationship(
        source_entity="Reasoner Planner",
        target_entity="Proxy Executor",
        description="Delegates tool execution.",
        keywords="delegates_to",
        score=0.85,
        source_chunk_ids=("common",),
        channels=("graph_expand",),
    )
    return {
        "planner architecture": RetrievalResult(
            query="planner architecture",
            chunks=(
                _chunk(
                    "common",
                    "The planner delegates tool execution to proxy agents.",
                    0.90,
                    "chunk_vector",
                ),
                _chunk(
                    "architecture",
                    "The architecture separates planning and execution.",
                    0.80,
                    "entity_source",
                ),
            ),
            entities=(shared_entity,),
            relationships=(shared_relationship,),
        ),
        "separation benefits": RetrievalResult(
            query="separation benefits",
            chunks=(
                _chunk(
                    "benefits",
                    "Isolation protects the planner context from tool noise.",
                    0.95,
                    "chunk_vector",
                ),
                _chunk(
                    "common",
                    "The planner delegates tool execution to proxy agents.",
                    0.70,
                    "graph_source",
                ),
            ),
            entities=(shared_entity,),
            relationships=(shared_relationship,),
        ),
    }


def _plan() -> ResearchPlan:
    return ResearchPlan(
        query="How does RP-ReAct separate planning and execution?",
        steps=(
            ResearchPlanStep(
                step_id="step-1",
                objective="Find the architecture.",
                search_query="planner architecture",
            ),
            ResearchPlanStep(
                step_id="step-2",
                objective="Find the benefits.",
                search_query="separation benefits",
            ),
        ),
        model_name="planner-model",
    )


def test_planner_retries_and_parses_bounded_plan() -> None:
    model = FakeModel(
        "planner-model",
        [
            '{"steps": []}',
            """<think>hidden</think>
            {"steps": [
              {"objective": "Find architecture", "query": "planner architecture"},
              {"objective": "Find benefits", "query": "separation benefits"}
            ]}""",
        ],
    )
    planner = ResearchPlanner(model, PlannerConfig(max_steps=2, retries=1))

    plan = asyncio.run(
        planner.plan("How does it work?", document_id="doc-paper")
    )

    assert len(model.requests) == 2
    assert "between 1 and 2" in model.requests[1].prompt
    assert "failed validation" in model.requests[1].prompt
    assert [step.step_id for step in plan.steps] == ["step-1", "step-2"]
    assert [step.search_query for step in plan.steps] == [
        "planner architecture",
        "separation benefits",
    ]
    assert plan.model_name == "planner-model"


def test_proxy_executor_fuses_multi_query_results_with_rrf() -> None:
    retriever = FakeRetriever(_results())
    executor = ProxyExecutor(
        retriever,
        ExecutorConfig(
            concurrency=2,
            max_chunks=3,
            max_entities=2,
            max_relationships=2,
            rrf_k=60,
        ),
    )

    retrieval, executions = asyncio.run(
        executor.execute(_plan(), document_id="doc-paper")
    )

    assert [chunk.chunk_id for chunk in retrieval.chunks] == [
        "common",
        "benefits",
        "architecture",
    ]
    assert retrieval.chunks[0].score == round(1 / 61 + 1 / 62, 6)
    assert retrieval.chunks[0].metadata["planner_queries"] == [
        "planner architecture",
        "separation benefits",
    ]
    assert "plan_retrieval" in retrieval.chunks[0].channels
    assert len(retrieval.entities) == 1
    assert retrieval.entities[0].score == round(2 / 61, 6)
    assert len(retrieval.relationships) == 1
    assert [execution.step_id for execution in executions] == [
        "step-1",
        "step-2",
    ]
    assert len(retriever.calls) == 2


def test_planner_reviews_evidence_and_creates_followup_step() -> None:
    model = FakeModel(
        "planner-model",
        [
            """{
              "decision": "retrieve",
              "reason": "The simple-task limitation is not supported.",
              "steps": [
                {
                  "objective": "Find the simple-task limitation.",
                  "query": "simple task planning overhead"
                }
              ]
            }"""
        ],
    )
    planner = ResearchPlanner(model)
    retrieval = ProxyExecutor(
        FakeRetriever(_results())
    ).merge_results(
        "How does it work?",
        [_results()["planner architecture"]],
    )

    review = asyncio.run(
        planner.review(
            "How does it work?",
            retrieval,
            ("planner architecture",),
            round_index=1,
            document_id="doc-paper",
        )
    )

    assert review.decision == "retrieve"
    assert review.added_steps[0].step_id == "replan-1-step-1"
    assert review.added_steps[0].search_query == (
        "simple task planning overhead"
    )
    assert review.evidence_chunk_ids == ("common", "architecture")
    assert "Already executed retrieval queries" in model.requests[0].prompt
    assert "chunk_id=common" in model.requests[0].prompt


def test_planner_rejects_duplicate_followup_query_and_retries() -> None:
    model = FakeModel(
        "planner-model",
        [
            """{
              "decision": "retrieve",
              "reason": "More evidence is needed.",
              "steps": [
                {"objective": "Repeat", "query": "planner architecture"}
              ]
            }""",
            """{
              "decision": "answer",
              "reason": "The existing evidence is sufficient.",
              "steps": []
            }""",
        ],
    )
    planner = ResearchPlanner(model, PlannerConfig(retries=1))
    retrieval = _results()["planner architecture"]

    review = asyncio.run(
        planner.review(
            "How does it work?",
            retrieval,
            ("planner architecture",),
            round_index=1,
        )
    )

    assert review.decision == "answer"
    assert len(model.requests) == 2
    assert "duplicate retrieval queries" in model.requests[1].prompt


def test_agent_plans_executes_and_answers_without_extra_retrieval() -> None:
    planner_model = FakeModel(
        "planner-model",
        [
            """{"steps": [
              {"objective": "Find architecture", "query": "planner architecture"},
              {"objective": "Find benefits", "query": "separation benefits"}
            ]}""",
            """{
              "decision": "answer",
              "reason": "Both architecture and benefits are supported.",
              "steps": []
            }""",
        ],
    )
    answer_model = FakeModel(
        "answer-model",
        [
            "RP-ReAct delegates execution to proxy agents [S1]. "
            "This protects planner context from tool noise [S2]."
        ],
    )
    retriever = FakeRetriever(_results())
    agent = RPReActAgent(
        ResearchPlanner(planner_model),
        ProxyExecutor(retriever),
        AnswerPipeline(retriever, answer_model),
    )

    result = asyncio.run(
        agent.run(
            "How does RP-ReAct separate planning and execution?",
            document_id="doc-paper",
        )
    )

    assert len(retriever.calls) == 2
    assert len(planner_model.requests) == 2
    assert len(answer_model.requests) == 1
    assert "The planner delegates tool execution" in answer_model.requests[0].prompt
    assert result.answer.cited_source_ids == ("S1", "S2")
    assert result.merged_chunk_count == 3
    assert result.answer.sources[0].chunk_id == "common"
    assert result.reviews[0].decision == "answer"
    assert result.stop_reason == "evidence_sufficient"
    payload = result.to_dict()
    assert payload["plan"]["steps"][0]["step_id"] == "step-1"
    assert payload["executions"][1]["tool_name"] == "retrieve_evidence"
    assert payload["merged_evidence_counts"]["chunks"] == 3


def test_agent_executes_followup_retrieval_after_review() -> None:
    planner_model = FakeModel(
        "planner-model",
        [
            """{"steps": [
              {"objective": "Find architecture", "query": "planner architecture"}
            ]}""",
            """{
              "decision": "retrieve",
              "reason": "Evidence about the benefit is missing.",
              "steps": [
                {"objective": "Find benefits", "query": "separation benefits"}
              ]
            }""",
        ],
    )
    answer_model = FakeModel(
        "answer-model",
        ["The architecture delegates execution [S1] and isolates noise [S2]."],
    )
    retriever = FakeRetriever(_results())
    agent = RPReActAgent(
        ResearchPlanner(planner_model),
        ProxyExecutor(retriever),
        AnswerPipeline(retriever, answer_model),
        AgentConfig(max_replans=1),
    )

    result = asyncio.run(
        agent.run("Explain the architecture and its benefit.")
    )

    assert [query for query, _ in retriever.calls] == [
        "planner architecture",
        "separation benefits",
    ]
    assert [execution.step_id for execution in result.executions] == [
        "step-1",
        "replan-1-step-1",
    ]
    assert result.reviews[0].decision == "retrieve"
    assert result.reviews[0].added_steps[0].search_query == (
        "separation benefits"
    )
    assert result.stop_reason == "replan_limit_reached"
    assert result.answer.cited_source_ids == ("S1", "S2")


def test_agent_uses_and_updates_persistent_session_memory(
    tmp_path: Path,
) -> None:
    memory = ConversationMemory(
        JsonConversationStore(tmp_path / "memory")
    )
    previous_source = AnswerSource(
        citation_id="S1",
        chunk_id="previous-chunk",
        source_id="previous-source",
        document_id="doc-paper",
        file_path="paper.pdf",
        page_start=2,
        page_end=2,
    )
    memory.remember(
        "research-1",
        "How does RP-ReAct separate planning and execution?",
        AnswerResult(
            query="How does RP-ReAct separate planning and execution?",
            answer="It uses a planner and proxy agents [S1].",
            sources=(previous_source,),
            cited_source_ids=("S1",),
            model_name="answer-model",
        ),
        document_id="doc-paper",
    )
    planner_model = FakeModel(
        "planner-model",
        [
            """{"steps": [
              {"objective": "Find its benefit", "query": "separation benefits"}
            ]}""",
            """{
              "decision": "answer",
              "reason": "The benefit is supported.",
              "steps": []
            }""",
        ],
    )
    answer_model = FakeModel(
        "answer-model",
        ["It protects planner context from tool noise [S1]."],
    )
    retriever = FakeRetriever(_results())
    agent = RPReActAgent(
        ResearchPlanner(planner_model),
        ProxyExecutor(retriever),
        AnswerPipeline(retriever, answer_model),
        memory=memory,
    )

    result = asyncio.run(
        agent.run(
            "Why is it beneficial?",
            document_id="doc-paper",
            session_id="research-1",
        )
    )

    assert "How does RP-ReAct separate" in planner_model.requests[0].prompt
    assert "It uses a planner and proxy agents" in (
        planner_model.requests[1].prompt
    )
    assert "How does RP-ReAct separate" in answer_model.requests[0].prompt
    assert result.session_id == "research-1"
    assert result.memory_turn_count == 2
    assert result.memory_path is not None
    assert Path(result.memory_path).is_file()
    session = memory.store.load("research-1")
    assert [turn.query for turn in session.turns] == [
        "How does RP-ReAct separate planning and execution?",
        "Why is it beneficial?",
    ]
