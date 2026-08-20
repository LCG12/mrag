from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from my_mrag import prompts
from my_mrag.models import AnalysisModel
from my_mrag.schemas import (
    AnalysisRequest,
    AnswerResult,
    AnswerSource,
    ContentType,
    RetrievalResult,
    RetrievedChunk,
)
from my_mrag.utils import stable_id


_SOURCE_CITATION_RE = re.compile(r"\[S(\d+)\]")
_GRAPH_CITATION_RE = re.compile(r"\[(?:E|R)\d+\]")


class EvidenceRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> RetrievalResult:
        ...


@dataclass(frozen=True)
class AnswerConfig:
    max_context_characters: int = 24000
    max_chunk_characters: int = 6000
    max_entities: int = 12
    max_relationships: int = 12
    citation_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_context_characters <= 0:
            raise ValueError("max_context_characters must be positive")
        if self.max_chunk_characters <= 0:
            raise ValueError("max_chunk_characters must be positive")
        if self.max_entities < 0 or self.max_relationships < 0:
            raise ValueError("graph evidence limits cannot be negative")
        if self.citation_retries < 0:
            raise ValueError("citation_retries cannot be negative")


class AnswerPipeline:
    """Generate a source-cited answer from hybrid retrieval evidence."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        model: AnalysisModel,
        config: AnswerConfig | None = None,
    ):
        self.retriever = retriever
        self.model = model
        self.config = config or AnswerConfig()

    async def answer(
        self,
        query: str,
        *,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> AnswerResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Answer query cannot be empty")

        retrieval = await self.retriever.retrieve(
            normalized_query,
            document_id=document_id,
        )
        return await self.answer_from_retrieval(
            normalized_query,
            retrieval,
            document_id=document_id,
            conversation_context=conversation_context,
        )

    async def answer_from_retrieval(
        self,
        query: str,
        retrieval: RetrievalResult,
        *,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> AnswerResult:
        """Generate an answer from evidence already gathered by an executor."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Answer query cannot be empty")

        request, sources = self.prepare_request(
            normalized_query,
            retrieval,
            document_id=document_id,
            conversation_context=conversation_context,
        )

        last_error: ValueError | None = None
        for attempt in range(self.config.citation_retries + 1):
            response = await self.model.complete(request)
            answer = self._strip_thinking_tags(response)
            try:
                cited_source_ids = self._validate_citations(answer, sources)
            except ValueError as error:
                last_error = error
                if attempt >= self.config.citation_retries:
                    break
                request = AnalysisRequest(
                    item_id=request.item_id,
                    document_id=request.document_id,
                    content_type=request.content_type,
                    system_prompt=request.system_prompt,
                    prompt=(
                        f"{request.prompt}\n\n"
                        "The previous answer failed citation validation: "
                        f"{error}. Rewrite the complete answer using only "
                        "the provided [S#] labels."
                    ),
                    context=request.context,
                )
                continue

            return AnswerResult(
                query=normalized_query,
                answer=answer,
                sources=sources,
                cited_source_ids=cited_source_ids,
                model_name=self.model.model_name,
                retrieved_entity_count=len(retrieval.entities),
                retrieved_relationship_count=len(retrieval.relationships),
            )

        raise RuntimeError(
            f"Answer generation failed citation validation: {last_error}"
        ) from last_error

    def prepare_request(
        self,
        query: str,
        retrieval: RetrievalResult,
        *,
        document_id: str | None = None,
        conversation_context: str = "",
    ) -> tuple[AnalysisRequest, tuple[AnswerSource, ...]]:
        source_budget = max(
            int(self.config.max_context_characters * 0.85),
            1,
        )
        graph_budget = self.config.max_context_characters - source_budget
        source_evidence, sources = self._build_source_evidence(
            retrieval.chunks,
            source_budget,
        )
        graph_evidence = self._build_graph_evidence(
            retrieval,
            graph_budget,
        )
        context = (
            f"Source evidence:\n{source_evidence}\n\n"
            f"Knowledge-graph hints:\n{graph_evidence}"
        )
        request = AnalysisRequest(
            item_id=stable_id(
                "answer",
                document_id or "all-documents",
                query,
            ),
            document_id=document_id or "all-documents",
            content_type=ContentType.TEXT,
            system_prompt=prompts.ANSWER_SYSTEM,
            prompt=prompts.ANSWER_PROMPT.format(
                query=query,
                conversation_context=conversation_context.strip() or "None",
                source_evidence=source_evidence,
                graph_evidence=graph_evidence,
            ),
            context=context,
        )
        return request, sources

    def _build_source_evidence(
        self,
        chunks: tuple[RetrievedChunk, ...],
        character_budget: int,
    ) -> tuple[str, tuple[AnswerSource, ...]]:
        blocks: list[str] = []
        sources: list[AnswerSource] = []
        used_characters = 0

        for chunk in chunks:
            citation_id = f"S{len(sources) + 1}"
            header = self._source_header(citation_id, chunk)
            separator_cost = 2 if blocks else 0
            available = (
                character_budget
                - used_characters
                - separator_cost
                - len(header)
                - 10
            )
            if available <= 0:
                break
            content = self._truncate_text(
                chunk.content.strip(),
                min(self.config.max_chunk_characters, available),
            )
            if not content:
                continue

            block = f"{header}\nContent:\n{content}"
            blocks.append(block)
            used_characters += len(block) + separator_cost
            sources.append(
                AnswerSource(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    document_id=chunk.document_id,
                    file_path=chunk.file_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                    content_type=chunk.content_type,
                    score=chunk.score,
                    asset_path=chunk.asset_path,
                    captions=chunk.captions,
                )
            )

        return "\n\n".join(blocks) or "None", tuple(sources)

    def _build_graph_evidence(
        self,
        retrieval: RetrievalResult,
        character_budget: int,
    ) -> str:
        if character_budget <= 0:
            return "None"

        lines: list[str] = []
        for index, entity in enumerate(
            retrieval.entities[: self.config.max_entities],
            start=1,
        ):
            description = self._truncate_text(entity.description, 300)
            lines.append(
                f"[E{index}] {entity.entity_name} "
                f"({entity.entity_type}): {description}"
            )

        for index, relationship in enumerate(
            retrieval.relationships[: self.config.max_relationships],
            start=1,
        ):
            description = self._truncate_text(
                relationship.description,
                300,
            )
            lines.append(
                f"[R{index}] {relationship.source_entity} -> "
                f"{relationship.target_entity} "
                f"({relationship.keywords}): {description}"
            )

        graph_evidence = "\n".join(lines) or "None"
        return self._truncate_text(graph_evidence, character_budget)

    @staticmethod
    def _source_header(citation_id: str, chunk: RetrievedChunk) -> str:
        document_name = Path(chunk.file_path).name or chunk.document_id or "Unknown"
        if chunk.page_start is None:
            pages = "Unknown"
        elif chunk.page_end in (None, chunk.page_start):
            pages = str(chunk.page_start)
        else:
            pages = f"{chunk.page_start}-{chunk.page_end}"
        section = " > ".join(chunk.section_path) or "None"
        captions = "; ".join(chunk.captions) or "None"
        return (
            f"[{citation_id}]\n"
            f"Document: {document_name}\n"
            f"PDF pages: {pages}\n"
            f"Section: {section}\n"
            f"Content type: {chunk.content_type}\n"
            f"Captions: {captions}"
        )

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        normalized = text.strip()
        if limit <= 0:
            return ""
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return normalized[:limit]
        candidate = normalized[: limit - 3]
        boundary = max(candidate.rfind(". "), candidate.rfind("\n"))
        if boundary >= int(len(candidate) * 0.7):
            candidate = candidate[: boundary + 1]
        return candidate.rstrip() + "..."

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        cleaned = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            text or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _validate_citations(
        answer: str,
        sources: tuple[AnswerSource, ...],
    ) -> tuple[str, ...]:
        if not answer:
            raise ValueError("model returned an empty answer")
        valid_ids = {source.citation_id for source in sources}
        cited_ids = tuple(
            dict.fromkeys(
                f"S{match}"
                for match in _SOURCE_CITATION_RE.findall(answer)
            )
        )
        if sources and not cited_ids:
            raise ValueError("answer contains no [S#] citation")
        graph_citations = tuple(
            dict.fromkeys(_GRAPH_CITATION_RE.findall(answer))
        )
        if graph_citations:
            raise ValueError(
                "answer cites graph hints instead of source evidence: "
                f"{list(graph_citations)}"
            )
        invalid_ids = [
            citation_id
            for citation_id in cited_ids
            if citation_id not in valid_ids
        ]
        if invalid_ids:
            raise ValueError(
                f"answer cites unknown sources: {invalid_ids}"
            )
        return cited_ids
