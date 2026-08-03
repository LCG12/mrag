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
