from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from typing import Any

from my_mrag import prompts
from my_mrag.answering import AnswerPipeline, EvidenceRetriever
from my_mrag.memory import ConversationMemory
from my_mrag.models import AnalysisModel
from my_mrag.schemas import (
    AgentResult,
    AnalysisRequest,
    ContentType,
    PlanReview,
    ResearchPlan,
    ResearchPlanStep,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    ToolExecution,
)
from my_mrag.utils import stable_id


@dataclass(frozen=True)
class PlannerConfig:
    max_steps: int = 3
    max_followup_steps: int = 1
    review_max_chunks: int = 8
    review_chunk_characters: int = 700
    retries: int = 1

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_followup_steps <= 0:
            raise ValueError("max_followup_steps must be positive")
        if self.review_max_chunks <= 0:
            raise ValueError("review_max_chunks must be positive")
        if self.review_chunk_characters <= 0:
            raise ValueError("review_chunk_characters must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")


@dataclass(frozen=True)
class ExecutorConfig:
    concurrency: int = 2
    max_chunks: int = 10
    max_entities: int = 24
    max_relationships: int = 24
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        if self.max_entities < 0 or self.max_relationships < 0:
            raise ValueError("graph evidence limits cannot be negative")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


@dataclass(frozen=True)
class AgentConfig:
    max_replans: int = 1

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError("max_replans cannot be negative")


class ResearchPlanner:
    """Create a bounded evidence-gathering plan without executing tools."""

    def __init__(
        self,
        model: AnalysisModel,
        config: PlannerConfig | None = None,
    ):
        self.model = model
        self.config = config or PlannerConfig()

    def prepare(
        self,
        query: str,
        *,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> AnalysisRequest:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Planning query cannot be empty")
        return AnalysisRequest(
            item_id=stable_id(
                "plan",
                document_id or "all-documents",
                normalized_query,
            ),
            document_id=document_id or "all-documents",
            content_type=ContentType.TEXT,
            system_prompt=prompts.RESEARCH_PLANNER_SYSTEM,
            prompt=prompts.RESEARCH_PLAN_PROMPT.format(
                query=normalized_query,
                conversation_context=(
                    conversation_context.strip() or "None"
                ),
                max_steps=self.config.max_steps,
            ),
            context=normalized_query,
        )

    async def plan(
        self,
        query: str,
        *,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> ResearchPlan:
        request = self.prepare(
            query,
            document_id=document_id,
            conversation_context=conversation_context,
        )
        last_error: ValueError | None = None
        for attempt in range(self.config.retries + 1):
            response = await self.model.complete(request)
            try:
                return self._parse_response(query.strip(), response)
            except ValueError as error:
                last_error = error
                if attempt >= self.config.retries:
                    break
                request = replace(
                    request,
                    prompt=(
                        f"{request.prompt}\n\n"
                        "The previous plan failed validation: "
                        f"{error}. Return a corrected complete JSON object."
                    ),
                )
        raise RuntimeError(f"Research planning failed: {last_error}") from last_error

    def prepare_review(
        self,
        query: str,
        retrieval: RetrievalResult,
        executed_queries: tuple[str, ...],
        *,
        round_index: int,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> AnalysisRequest:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Review query cannot be empty")
        if round_index <= 0:
            raise ValueError("round_index must be positive")
        executed = "\n".join(
            f"- {value}" for value in executed_queries
        ) or "None"
        evidence_summary = self._review_evidence_summary(retrieval)
        return AnalysisRequest(
            item_id=stable_id(
                "review",
                document_id or "all-documents",
                normalized_query,
                round_index,
            ),
            document_id=document_id or "all-documents",
            content_type=ContentType.TEXT,
            system_prompt=prompts.RESEARCH_REVIEW_SYSTEM,
            prompt=prompts.RESEARCH_REVIEW_PROMPT.format(
                query=normalized_query,
                conversation_context=(
                    conversation_context.strip() or "None"
                ),
                executed_queries=executed,
                evidence_summary=evidence_summary,
                max_followup_steps=self.config.max_followup_steps,
            ),
            context=evidence_summary,
        )

    async def review(
        self,
        query: str,
        retrieval: RetrievalResult,
        executed_queries: tuple[str, ...],
        *,
        round_index: int,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> PlanReview:
        request = self.prepare_review(
            query,
            retrieval,
            executed_queries,
            round_index=round_index,
            document_id=document_id,
            conversation_context=conversation_context,
        )
        last_error: ValueError | None = None
        for attempt in range(self.config.retries + 1):
            response = await self.model.complete(request)
            try:
                return self._parse_review(
                    response,
                    retrieval,
                    executed_queries,
                    round_index=round_index,
                )
            except ValueError as error:
                last_error = error
                if attempt >= self.config.retries:
                    break
                request = replace(
                    request,
                    prompt=(
                        f"{request.prompt}\n\n"
                        "The previous review failed validation: "
                        f"{error}. Return a corrected complete JSON object."
                    ),
                )
        raise RuntimeError(f"Evidence review failed: {last_error}") from last_error

    def _parse_response(self, query: str, response: str) -> ResearchPlan:
        cleaned = self._strip_thinking_tags(response)
        payload = self._parse_json_object(cleaned)
        if payload is None:
            raise ValueError("model response does not contain a JSON object")
        steps = self._parse_steps(
            payload.get("steps"),
            max_steps=self.config.max_steps,
            step_prefix="step",
        )

        return ResearchPlan(
            query=query,
            steps=tuple(steps),
            model_name=self.model.model_name,
        )

    def _parse_review(
        self,
        response: str,
        retrieval: RetrievalResult,
        executed_queries: tuple[str, ...],
        *,
        round_index: int,
    ) -> PlanReview:
        cleaned = self._strip_thinking_tags(response)
        payload = self._parse_json_object(cleaned)
        if payload is None:
            raise ValueError("model response does not contain a JSON object")
        decision = str(payload.get("decision") or "").strip().casefold()
        if decision not in {"answer", "retrieve"}:
            raise ValueError("decision must be 'answer' or 'retrieve'")
        reason = self._required_text(payload, "reason")
        raw_steps = payload.get("steps")
        if decision == "answer":
            if raw_steps != []:
                raise ValueError("answer decision must contain an empty steps array")
            added_steps: tuple[ResearchPlanStep, ...] = ()
        else:
            added_steps = self._parse_steps(
                raw_steps,
                max_steps=self.config.max_followup_steps,
                step_prefix=f"replan-{round_index}-step",
                existing_queries=executed_queries,
            )
        return PlanReview(
            round_index=round_index,
            decision=decision,
            reason=reason,
            evidence_chunk_ids=tuple(
                chunk.chunk_id
                for chunk in retrieval.chunks[: self.config.review_max_chunks]
            ),
            added_steps=added_steps,
        )

    def _parse_steps(
        self,
        raw_steps: Any,
        *,
        max_steps: int,
        step_prefix: str,
        existing_queries: tuple[str, ...] = (),
    ) -> tuple[ResearchPlanStep, ...]:
        if not isinstance(raw_steps, list):
            raise ValueError("steps must be a JSON array")
        if not 1 <= len(raw_steps) <= max_steps:
            raise ValueError(
                f"steps must contain between 1 and {max_steps} items"
            )

        seen_queries = {
            self._text_key(value) for value in existing_queries
        }
        steps: list[ResearchPlanStep] = []
        for index, value in enumerate(raw_steps, start=1):
            if not isinstance(value, dict):
                raise ValueError("each plan step must be a JSON object")
            objective = self._required_text(value, "objective")
            search_query = self._required_text(value, "query")
            query_key = self._text_key(search_query)
            if query_key in seen_queries:
                raise ValueError("plan contains duplicate retrieval queries")
            seen_queries.add(query_key)
            steps.append(
                ResearchPlanStep(
                    step_id=f"{step_prefix}-{index}",
                    objective=objective,
                    search_query=search_query,
                )
            )
        return tuple(steps)

    def _review_evidence_summary(self, retrieval: RetrievalResult) -> str:
        lines = [
            "Evidence counts: "
            f"chunks={len(retrieval.chunks)}, "
            f"entities={len(retrieval.entities)}, "
            f"relationships={len(retrieval.relationships)}"
        ]
        chunks = retrieval.chunks[: self.config.review_max_chunks]
        if not chunks:
            lines.append("No source chunks were retrieved.")
        for index, chunk in enumerate(chunks, start=1):
            if chunk.page_start is None:
                pages = "unknown"
            elif chunk.page_end in (None, chunk.page_start):
                pages = str(chunk.page_start)
            else:
                pages = f"{chunk.page_start}-{chunk.page_end}"
            section = " > ".join(chunk.section_path) or "None"
            content = self._truncate_text(
                chunk.content,
                self.config.review_chunk_characters,
            )
            lines.append(
                f"[C{index}] chunk_id={chunk.chunk_id}; pages={pages}; "
                f"section={section}; modality={chunk.content_type}\n{content}"
            )
        if retrieval.entities:
            entity_names = ", ".join(
                entity.entity_name for entity in retrieval.entities[:10]
            )
            lines.append(f"Graph navigation hints: {entity_names}")
        return "\n\n".join(lines)

    @staticmethod
    def _required_text(value: dict[str, Any], field_name: str) -> str:
        text = str(value.get(field_name) or "").strip()
        if not text:
            raise ValueError(f"plan step {field_name} cannot be empty")
        return text

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _text_key(value: str) -> str:
        return " ".join(value.split()).casefold()

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        return re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            text or "",
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None


class ProxyExecutor:
    """Execute retrieval steps and fuse their observations for the reasoner."""

    tool_name = "retrieve_evidence"

    def __init__(
        self,
        retriever: EvidenceRetriever,
        config: ExecutorConfig | None = None,
    ):
        self.retriever = retriever
        self.config = config or ExecutorConfig()

    async def execute(
        self,
        plan: ResearchPlan,
        *,
        document_id: str | None = None,
    ) -> tuple[RetrievalResult, tuple[ToolExecution, ...]]:
        results, executions = await self.execute_results(
            plan,
            document_id=document_id,
        )
        return self.merge_results(plan.query, list(results)), executions

    async def execute_results(
        self,
        plan: ResearchPlan,
        *,
        document_id: str | None = None,
    ) -> tuple[tuple[RetrievalResult, ...], tuple[ToolExecution, ...]]:
        """Execute a batch while preserving each query's original ranking."""

        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def run_step(
            step: ResearchPlanStep,
        ) -> tuple[RetrievalResult, ToolExecution]:
            async with semaphore:
                result = await self.retriever.retrieve(
                    step.search_query,
                    document_id=document_id,
                )
            return result, ToolExecution(
                step_id=step.step_id,
                objective=step.objective,
                tool_name=self.tool_name,
                tool_input=step.search_query,
                retrieved_chunk_ids=tuple(
                    chunk.chunk_id for chunk in result.chunks
                ),
                retrieved_entity_count=len(result.entities),
                retrieved_relationship_count=len(result.relationships),
            )

        observations = await asyncio.gather(
            *(run_step(step) for step in plan.steps)
        )
        results = tuple(result for result, _ in observations)
        executions = tuple(execution for _, execution in observations)
        return results, executions

    def merge_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            chunks=self._merge_chunks(results),
            entities=self._merge_entities(results),
            relationships=self._merge_relationships(results),
        )

    def _merge_chunks(
        self,
        results: list[RetrievalResult],
    ) -> tuple[RetrievedChunk, ...]:
        states: dict[str, dict[str, Any]] = {}
        for result in results:
            for rank, chunk in enumerate(result.chunks, start=1):
                state = states.setdefault(
                    chunk.chunk_id,
                    {
                        "item": chunk,
                        "rrf": 0.0,
                        "best_score": chunk.score,
                        "channels": [],
                        "queries": [],
                    },
                )
                state["rrf"] += self._rrf(rank)
                state["best_score"] = max(state["best_score"], chunk.score)
                state["channels"].extend(chunk.channels)
                state["queries"].append(result.query)
                if chunk.score > state["item"].score:
                    state["item"] = chunk

        merged: list[RetrievedChunk] = []
        for state in self._sorted_states(states):
            chunk = state["item"]
            metadata = dict(chunk.metadata)
            metadata["planner_queries"] = list(
                dict.fromkeys(state["queries"])
            )
            metadata["best_retrieval_score"] = state["best_score"]
            merged.append(
                replace(
                    chunk,
                    score=round(state["rrf"], 6),
                    channels=tuple(
                        dict.fromkeys(
                            [*state["channels"], "plan_retrieval"]
                        )
                    ),
                    metadata=metadata,
                )
            )
        return tuple(merged[: self.config.max_chunks])

    def _merge_entities(
        self,
        results: list[RetrievalResult],
    ) -> tuple[RetrievedEntity, ...]:
        states: dict[str, dict[str, Any]] = {}
        for result in results:
            for rank, entity in enumerate(result.entities, start=1):
                key = self._text_key(entity.entity_name)
                state = states.setdefault(
                    key,
                    {
                        "item": entity,
                        "rrf": 0.0,
                        "best_score": entity.score,
                        "source_ids": [],
                        "channels": [],
                    },
                )
                state["rrf"] += self._rrf(rank)
                state["best_score"] = max(state["best_score"], entity.score)
                state["source_ids"].extend(entity.source_chunk_ids)
                state["channels"].extend(entity.channels)
                if len(entity.description) > len(state["item"].description):
                    state["item"] = entity

        merged: list[RetrievedEntity] = []
        for state in self._sorted_states(states):
            entity = state["item"]
            merged.append(
                replace(
                    entity,
                    score=round(state["rrf"], 6),
                    source_chunk_ids=tuple(
                        dict.fromkeys(state["source_ids"])
                    ),
                    channels=tuple(
                        dict.fromkeys(
                            [*state["channels"], "plan_retrieval"]
                        )
                    ),
                )
            )
        return tuple(merged[: self.config.max_entities])

    def _merge_relationships(
        self,
        results: list[RetrievalResult],
    ) -> tuple[RetrievedRelationship, ...]:
        states: dict[tuple[str, str], dict[str, Any]] = {}
        for result in results:
            for rank, relationship in enumerate(
                result.relationships,
                start=1,
            ):
                key = (
                    self._text_key(relationship.source_entity),
                    self._text_key(relationship.target_entity),
                )
                state = states.setdefault(
                    key,
                    {
                        "item": relationship,
                        "rrf": 0.0,
                        "best_score": relationship.score,
                        "source_ids": [],
                        "channels": [],
                    },
                )
                state["rrf"] += self._rrf(rank)
                state["best_score"] = max(
                    state["best_score"],
                    relationship.score,
                )
                state["source_ids"].extend(relationship.source_chunk_ids)
                state["channels"].extend(relationship.channels)
                if len(relationship.description) > len(
                    state["item"].description
                ):
                    state["item"] = relationship

        merged: list[RetrievedRelationship] = []
        for state in self._sorted_states(states):
            relationship = state["item"]
            merged.append(
                replace(
                    relationship,
                    score=round(state["rrf"], 6),
                    source_chunk_ids=tuple(
                        dict.fromkeys(state["source_ids"])
                    ),
                    channels=tuple(
                        dict.fromkeys(
                            [*state["channels"], "plan_retrieval"]
                        )
                    ),
                )
            )
        return tuple(merged[: self.config.max_relationships])

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.config.rrf_k + rank)

    @staticmethod
    def _sorted_states(
        states: dict[Any, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            states.values(),
            key=lambda state: (
                state["rrf"],
                state["best_score"],
            ),
            reverse=True,
        )

    @staticmethod
    def _text_key(value: str) -> str:
        return " ".join(value.split()).casefold()


class RPReActAgent:
    """Coordinate high-level planning, proxy execution, and final synthesis."""

    def __init__(
        self,
        planner: ResearchPlanner,
        executor: ProxyExecutor,
        answer_pipeline: AnswerPipeline,
        config: AgentConfig | None = None,
        memory: ConversationMemory | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.answer_pipeline = answer_pipeline
        self.config = config or AgentConfig()
        self.memory = memory

    async def run(
        self,
        query: str,
        *,
        document_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Agent query cannot be empty")
        if session_id and self.memory is None:
            raise ValueError("Agent memory is required when session_id is set")
        conversation_context = (
            self.memory.context(session_id)
            if session_id and self.memory is not None
            else ""
        )

        plan = await self.planner.plan(
            normalized_query,
            document_id=document_id,
            conversation_context=conversation_context,
        )
        initial_results, initial_executions = await self.executor.execute_results(
            plan,
            document_id=document_id,
        )
        all_results = list(initial_results)
        executions = list(initial_executions)
        executed_queries = [step.search_query for step in plan.steps]
        reviews: list[PlanReview] = []
        stop_reason = (
            "review_disabled"
            if self.config.max_replans == 0
            else "replan_limit_reached"
        )

        for round_index in range(1, self.config.max_replans + 1):
            retrieval = self.executor.merge_results(
                normalized_query,
                all_results,
            )
            review = await self.planner.review(
                normalized_query,
                retrieval,
                tuple(executed_queries),
                round_index=round_index,
                document_id=document_id,
                conversation_context=conversation_context,
            )
            reviews.append(review)
            if review.decision == "answer":
                stop_reason = "evidence_sufficient"
                break

            followup_plan = ResearchPlan(
                query=normalized_query,
                steps=review.added_steps,
                model_name=self.planner.model.model_name,
            )
            followup_results, followup_executions = (
                await self.executor.execute_results(
                    followup_plan,
                    document_id=document_id,
                )
            )
            all_results.extend(followup_results)
            executions.extend(followup_executions)
            executed_queries.extend(
                step.search_query for step in review.added_steps
            )

        retrieval = self.executor.merge_results(
            normalized_query,
            all_results,
        )
        answer = await self.answer_pipeline.answer_from_retrieval(
            normalized_query,
            retrieval,
            document_id=document_id,
            conversation_context=conversation_context,
        )
        result = AgentResult(
            query=normalized_query,
            plan=plan,
            executions=tuple(executions),
            answer=answer,
            reviews=tuple(reviews),
            stop_reason=stop_reason,
            merged_chunk_count=len(retrieval.chunks),
            merged_entity_count=len(retrieval.entities),
            merged_relationship_count=len(retrieval.relationships),
        )
        if session_id and self.memory is not None:
            session, stored_path = self.memory.remember(
                session_id,
                normalized_query,
                answer,
                document_id=document_id,
            )
            result = replace(
                result,
                session_id=session.session_id,
                memory_turn_count=len(session.turns),
                memory_path=str(stored_path),
            )
        return result
