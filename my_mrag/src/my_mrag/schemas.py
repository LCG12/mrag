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
