from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_value(cls, value: Any) -> "BoundingBox | None":
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError(f"Bounding box must contain four values: {value!r}")
        return cls(*(round(float(number), 3) for number in value))


@dataclass
class ContentItem:
    """One normalized text or multimodal item from a source document."""

    item_id: str
    document_id: str
    type: ContentType
    page_idx: int
    order_idx: int
    text: str = ""
    bbox: BoundingBox | None = None
    asset_path: str | None = None
    captions: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContentItem":
        data = dict(payload)
        data["type"] = ContentType(data["type"])
        bbox = data.get("bbox")
        if bbox:
            data["bbox"] = BoundingBox(**bbox)
        return cls(**data)


@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    parser: str
    page_count: int
    items: list[ContentItem]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "parser": self.parser,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParsedDocument":
        return cls(
            document_id=payload["document_id"],
            source_path=payload["source_path"],
            parser=payload["parser"],
            page_count=int(payload["page_count"]),
            metadata=dict(payload.get("metadata") or {}),
            items=[
                ContentItem.from_dict(item)
                for item in payload.get("items", [])
            ],
        )

    def count_by_type(self) -> dict[str, int]:
        counts = {content_type.value: 0 for content_type in ContentType}
        for item in self.items:
            counts[item.type.value] += 1
        return counts


@dataclass(frozen=True)
class TextChunk:
    """One retrieval-ready chunk derived from ordered text items."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_start: int
    page_end: int
    source_order_start: int
    source_item_ids: tuple[str, ...]
    section_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def index_text(self) -> str:
        if not self.section_path:
            return self.text
        return (
            f"Section: {' > '.join(self.section_path)}\n\n"
            f"{self.text}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_item_ids"] = list(self.source_item_ids)
        payload["section_path"] = list(self.section_path)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TextChunk":
        data = dict(payload)
        data["source_item_ids"] = tuple(data.get("source_item_ids") or ())
        data["section_path"] = tuple(data.get("section_path") or ())
        data["metadata"] = dict(data.get("metadata") or {})
        return cls(**data)


@dataclass(frozen=True)
class KnowledgeEntity:
    """One canonical entity extracted from a text chunk."""

    entity_name: str
    entity_type: str
    description: str

    def __post_init__(self) -> None:
        if not self.entity_name.strip():
            raise ValueError("Knowledge entity name cannot be empty")
        if not self.entity_type.strip():
            raise ValueError("Knowledge entity type cannot be empty")
        if not self.description.strip():
            raise ValueError("Knowledge entity description cannot be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeEntity":
        return cls(
            entity_name=str(payload["entity_name"]).strip(),
            entity_type=str(payload["entity_type"]).strip().lower(),
            description=str(payload["description"]).strip(),
        )


@dataclass(frozen=True)
class KnowledgeRelationship:
    """One evidence-backed relation between two extracted entities."""

    source_entity: str
    target_entity: str
    description: str
    keywords: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_entity.strip() or not self.target_entity.strip():
            raise ValueError("Knowledge relationship endpoints cannot be empty")
        if not self.description.strip():
            raise ValueError("Knowledge relationship description cannot be empty")
        if self.weight <= 0:
            raise ValueError("Knowledge relationship weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keywords"] = list(self.keywords)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeRelationship":
        raw_keywords = payload.get("keywords") or ()
        if isinstance(raw_keywords, str):
            raw_keywords = raw_keywords.split(",")
        return cls(
            source_entity=str(payload["source_entity"]).strip(),
            target_entity=str(payload["target_entity"]).strip(),
            description=str(payload["description"]).strip(),
            keywords=tuple(
                str(keyword).strip()
                for keyword in raw_keywords
                if str(keyword).strip()
            ),
            weight=float(payload.get("weight") or 1.0),
        )


@dataclass(frozen=True)
class ChunkKnowledge:
    """Structured entities and relationships extracted from one text chunk."""

    chunk_id: str
    document_id: str
    chunk_index: int
    entities: tuple[KnowledgeEntity, ...] = ()
    relationships: tuple[KnowledgeRelationship, ...] = ()
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "entities": [asdict(entity) for entity in self.entities],
            "relationships": [
                relationship.to_dict()
                for relationship in self.relationships
            ],
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkKnowledge":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            chunk_index=int(payload["chunk_index"]),
            entities=tuple(
                KnowledgeEntity.from_dict(entity)
                for entity in payload.get("entities", [])
            ),
            relationships=tuple(
                KnowledgeRelationship.from_dict(relationship)
                for relationship in payload.get("relationships", [])
            ),
            model_name=str(payload.get("model_name") or ""),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """One source-resolved chunk returned by hybrid retrieval."""

    chunk_id: str
    score: float
    content: str
    document_id: str = ""
    source_id: str = ""
    file_path: str = ""
    chunk_order_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    content_type: str = "text"
    asset_path: str | None = None
    captions: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["section_path"] = list(self.section_path)
        payload["captions"] = list(self.captions)
        payload["channels"] = list(self.channels)
        return payload


@dataclass(frozen=True)
class RetrievedEntity:
    entity_name: str
    entity_type: str
    description: str
    score: float
    source_chunk_ids: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_chunk_ids"] = list(self.source_chunk_ids)
        payload["channels"] = list(self.channels)
        return payload


@dataclass(frozen=True)
class RetrievedRelationship:
    source_entity: str
    target_entity: str
    description: str
    keywords: str
    score: float
    weight: float = 1.0
    source_chunk_ids: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_chunk_ids"] = list(self.source_chunk_ids)
        payload["channels"] = list(self.channels)
        return payload


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: tuple[RetrievedChunk, ...] = ()
    entities: tuple[RetrievedEntity, ...] = ()
    relationships: tuple[RetrievedRelationship, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [
                relationship.to_dict()
                for relationship in self.relationships
            ],
        }


@dataclass(frozen=True)
class AnswerSource:
    """One citation target exposed with a generated answer."""

    citation_id: str
    chunk_id: str
    source_id: str
    document_id: str
    file_path: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...] = ()
    content_type: str = "text"
    score: float = 0.0
    asset_path: str | None = None
    captions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["section_path"] = list(self.section_path)
        payload["captions"] = list(self.captions)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerSource":
        data = dict(payload)
        data["section_path"] = tuple(data.get("section_path", ()))
        data["captions"] = tuple(data.get("captions", ()))
        return cls(**data)


@dataclass(frozen=True)
class AnswerResult:
    """A grounded model answer and its source citation map."""

    query: str
    answer: str
    sources: tuple[AnswerSource, ...]
    cited_source_ids: tuple[str, ...]
    model_name: str
    retrieved_entity_count: int = 0
    retrieved_relationship_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "model_name": self.model_name,
            "cited_source_ids": list(self.cited_source_ids),
            "sources": [source.to_dict() for source in self.sources],
            "evidence_counts": {
                "sources": len(self.sources),
                "entities": self.retrieved_entity_count,
                "relationships": self.retrieved_relationship_count,
            },
        }


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user/assistant exchange stored in session memory."""

    turn_id: str
    query: str
    answer: str
    document_id: str | None
    created_at: str
    cited_sources: tuple[AnswerSource, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "answer": self.answer,
            "document_id": self.document_id,
            "created_at": self.created_at,
            "cited_sources": [
                source.to_dict() for source in self.cited_sources
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationTurn":
        return cls(
            turn_id=str(payload["turn_id"]),
            query=str(payload["query"]),
            answer=str(payload["answer"]),
            document_id=(
                str(payload["document_id"])
                if payload.get("document_id") is not None
                else None
            ),
            created_at=str(payload.get("created_at") or ""),
            cited_sources=tuple(
                AnswerSource.from_dict(source)
                for source in payload.get("cited_sources", [])
            ),
        )


@dataclass(frozen=True)
class ConversationSession:
    """Ordered turns associated with one user-selected session ID."""

    session_id: str
    turns: tuple[ConversationTurn, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationSession":
        return cls(
            session_id=str(payload["session_id"]),
            turns=tuple(
                ConversationTurn.from_dict(turn)
                for turn in payload.get("turns", [])
            ),
        )


@dataclass(frozen=True)
class ResearchPlanStep:
    """One evidence-gathering objective produced by the reasoner planner."""

    step_id: str
    objective: str
    search_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchPlan:
    """A bounded high-level plan for answering one research question."""

    query: str
    steps: tuple[ResearchPlanStep, ...]
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "model_name": self.model_name,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class PlanReview:
    """One evidence-sufficiency decision made after tool observations."""

    round_index: int
    decision: str
    reason: str
    evidence_chunk_ids: tuple[str, ...] = ()
    added_steps: tuple[ResearchPlanStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "decision": self.decision,
            "reason": self.reason,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "added_steps": [step.to_dict() for step in self.added_steps],
        }


@dataclass(frozen=True)
class ToolExecution:
    """Observable result of one proxy-executor tool call."""

    step_id: str
    objective: str
    tool_name: str
    tool_input: str
    retrieved_chunk_ids: tuple[str, ...] = ()
    retrieved_entity_count: int = 0
    retrieved_relationship_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_chunk_ids"] = list(self.retrieved_chunk_ids)
        return payload


@dataclass(frozen=True)
class AgentResult:
    """RP-ReAct plan, execution trace, and final grounded answer."""

    query: str
    plan: ResearchPlan
    executions: tuple[ToolExecution, ...]
    answer: AnswerResult
    reviews: tuple[PlanReview, ...] = ()
    stop_reason: str = "completed"
    session_id: str | None = None
    memory_turn_count: int = 0
    memory_path: str | None = None
    merged_chunk_count: int = 0
    merged_entity_count: int = 0
    merged_relationship_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.to_dict(),
            "executions": [execution.to_dict() for execution in self.executions],
            "reviews": [review.to_dict() for review in self.reviews],
            "stop_reason": self.stop_reason,
            "memory": (
                {
                    "session_id": self.session_id,
                    "turn_count": self.memory_turn_count,
                    "stored_path": self.memory_path,
                }
                if self.session_id
                else None
            ),
            "merged_evidence_counts": {
                "chunks": self.merged_chunk_count,
                "entities": self.merged_entity_count,
                "relationships": self.merged_relationship_count,
            },
            "answer": self.answer.to_dict(),
        }


@dataclass(frozen=True)
class EntityInfo:
    """Entity generated from one multimodal item."""

    entity_name: str
    entity_type: str
    summary: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityInfo":
        return cls(
            entity_name=str(payload["entity_name"]).strip(),
            entity_type=str(payload["entity_type"]).strip(),
            summary=str(payload["summary"]).strip(),
        )


@dataclass(frozen=True)
class AnalysisRequest:
    """Complete model input prepared by a modality-specific processor."""

    item_id: str
    document_id: str
    content_type: ContentType
    system_prompt: str
    prompt: str
    context: str = ""
    image_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_type"] = self.content_type.value
        payload["image_paths"] = list(self.image_paths)
        return payload


@dataclass(frozen=True)
class ModalAnalysis:
    """Normalized output produced by a multimodal processor."""

    item_id: str
    document_id: str
    content_type: ContentType
    detailed_description: str
    entity_info: EntityInfo
    context: str
    chunk_text: str
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_type"] = self.content_type.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalAnalysis":
        data = dict(payload)
        data["content_type"] = ContentType(data["content_type"])
        data["entity_info"] = EntityInfo.from_dict(data["entity_info"])
        return cls(**data)
