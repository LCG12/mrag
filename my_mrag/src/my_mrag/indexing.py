from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from my_mrag.schemas import (
    ChunkKnowledge,
    ModalAnalysis,
    ParsedDocument,
    TextChunk,
)


class CustomKGStore(Protocol):
    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str | None = None,
    ) -> None:
        ...


@dataclass(frozen=True)
class IndexingReport:
    document_id: str
    chunk_count: int
    text_chunk_count: int
    multimodal_chunk_count: int
    knowledge_chunk_count: int
    text_entity_count: int
    text_relationship_count: int
    entity_count: int
    relationship_count: int
    document_entity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LightRAGIndexer:
    """Convert our multimodal analyses into LightRAG's custom-KG format."""

    def __init__(self, store: CustomKGStore):
        self.store = store

    def build_payload(
        self,
        document: ParsedDocument,
        analyses: list[ModalAnalysis],
        text_chunks: list[TextChunk] | None = None,
        knowledge: list[ChunkKnowledge] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        text_chunks = list(text_chunks or [])
        knowledge = list(knowledge or [])
        if not analyses and not text_chunks:
            return {"chunks": [], "entities": [], "relationships": []}

        items_by_id = {item.item_id: item for item in document.items}
        seen_item_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()
        entity_names: set[str] = set()
        chunks: list[dict[str, Any]] = []
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        document_title = str(
            document.metadata.get("title")
            or Path(document.source_path).stem
            or document.document_id
        ).strip()
        document_entity = self.document_entity_name(document.document_id)
        file_path = document.source_path

        for text_chunk in text_chunks:
            self._validate_text_chunk(
                document,
                text_chunk,
                items_by_id,
                seen_chunk_ids,
            )
            chunks.append(
                {
                    "content": text_chunk.index_text(),
                    "source_id": text_chunk.chunk_id,
                    "chunk_order_index": text_chunk.source_order_start,
                    "file_path": file_path,
                }
            )

        for analysis in analyses:
            self._validate_analysis(
                document,
                analysis,
                items_by_id,
                seen_item_ids,
            )
            item = items_by_id[analysis.item_id]
            source_id = item.item_id
            entity_name = self._unique_entity_name(
                analysis,
                entity_names,
            )
            entity_names.add(entity_name)

            chunks.append(
                {
                    "content": analysis.chunk_text,
                    "source_id": source_id,
                    "chunk_order_index": item.order_idx,
                    "file_path": file_path,
                }
            )
            entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": analysis.content_type.value,
                    "description": analysis.entity_info.summary,
                    "source_id": source_id,
                    "file_path": file_path,
                }
            )
            relationships.append(
                {
                    "src_id": entity_name,
                    "tgt_id": document_entity,
                    "description": (
                        f"{entity_name} is multimodal content from "
                        f"the source document {document_title}."
                    ),
                    "keywords": (
                        "belongs_to,source_document,multimodal_content"
                    ),
                    "weight": 10.0,
                    "source_id": source_id,
                    "file_path": file_path,
                }
            )

        text_entities, text_relationships = self._build_text_knowledge(
            document,
            text_chunks,
            knowledge,
            file_path,
        )
        entities.extend(text_entities)
        relationships.extend(text_relationships)

        chunks.sort(key=lambda chunk: int(chunk["chunk_order_index"]))

        # LightRAG requires every entity source_id to map to a custom chunk.
        first_source_id = str(chunks[0]["source_id"])
        entities.append(
            {
                "entity_name": document_entity,
                "entity_type": "document",
                "description": (
                    f"Source research document: {document_title}. "
                    f"Document ID: {document.document_id}."
                ),
                "source_id": first_source_id,
                "file_path": file_path,
            }
        )

        return {
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
        }

    async def index(
        self,
        document: ParsedDocument,
        analyses: list[ModalAnalysis],
        text_chunks: list[TextChunk] | None = None,
        knowledge: list[ChunkKnowledge] | None = None,
    ) -> IndexingReport:
        normalized_text_chunks = list(text_chunks or [])
        normalized_knowledge = list(knowledge or [])
        payload = self.build_payload(
            document,
            analyses,
            normalized_text_chunks,
            normalized_knowledge,
        )
        if payload["chunks"]:
            await self.store.ainsert_custom_kg(
                payload,
                full_doc_id=document.document_id,
            )
        return IndexingReport(
            document_id=document.document_id,
            chunk_count=len(payload["chunks"]),
            text_chunk_count=len(normalized_text_chunks),
            multimodal_chunk_count=len(analyses),
            knowledge_chunk_count=len(normalized_knowledge),
            text_entity_count=sum(
                len(extraction.entities)
                for extraction in normalized_knowledge
            ),
            text_relationship_count=sum(
                len(extraction.relationships)
                for extraction in normalized_knowledge
            ),
            entity_count=len(payload["entities"]),
            relationship_count=len(payload["relationships"]),
            document_entity=(
                self.document_entity_name(document.document_id)
                if payload["chunks"]
                else ""
            ),
        )

    @staticmethod
    def document_entity_name(document_id: str) -> str:
        return f"DOCUMENT::{document_id}"

    @classmethod
    def _build_text_knowledge(
        cls,
        document: ParsedDocument,
        text_chunks: list[TextChunk],
        knowledge: list[ChunkKnowledge],
        file_path: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not knowledge:
            return [], []

        chunks_by_id = {chunk.chunk_id: chunk for chunk in text_chunks}
        seen_extractions: set[str] = set()
        entity_records: dict[str, dict[str, Any]] = {}

        for extraction in knowledge:
            cls._validate_knowledge_extraction(
                document,
                extraction,
                chunks_by_id,
                seen_extractions,
            )
            for entity in extraction.entities:
                key = cls._entity_key(entity.entity_name)
                record = entity_records.setdefault(
                    key,
                    {
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "descriptions": [],
                        "source_id": extraction.chunk_id,
                    },
                )
                if entity.description not in record["descriptions"]:
                    record["descriptions"].append(entity.description)

        relationship_records: dict[tuple[str, str], dict[str, Any]] = {}
        for extraction in knowledge:
            for relationship in extraction.relationships:
                source_key = cls._entity_key(relationship.source_entity)
                target_key = cls._entity_key(relationship.target_entity)
                if source_key not in entity_records or target_key not in entity_records:
                    raise ValueError(
                        "Knowledge relationship references an unknown entity: "
                        f"{relationship.source_entity!r} -> "
                        f"{relationship.target_entity!r}"
                    )
                source = entity_records[source_key]["entity_name"]
                target = entity_records[target_key]["entity_name"]
                key = tuple(sorted((source_key, target_key)))
                record = relationship_records.setdefault(
                    key,
                    {
                        "src_id": source,
                        "tgt_id": target,
                        "descriptions": [],
                        "keywords": [],
                        "weight": relationship.weight,
                        "source_id": extraction.chunk_id,
                    },
                )
                if relationship.description not in record["descriptions"]:
                    record["descriptions"].append(relationship.description)
                for keyword in relationship.keywords:
                    if keyword not in record["keywords"]:
                        record["keywords"].append(keyword)
                record["weight"] = max(record["weight"], relationship.weight)

        entities = [
            {
                "entity_name": record["entity_name"],
                "entity_type": record["entity_type"],
                "description": cls._merge_descriptions(record["descriptions"]),
                "source_id": record["source_id"],
                "file_path": file_path,
            }
            for record in entity_records.values()
        ]
        relationships = [
            {
                "src_id": record["src_id"],
                "tgt_id": record["tgt_id"],
                "description": cls._merge_descriptions(record["descriptions"]),
                "keywords": ",".join(record["keywords"] or ["related_to"]),
                "weight": record["weight"],
                "source_id": record["source_id"],
                "file_path": file_path,
            }
            for record in relationship_records.values()
        ]
        return entities, relationships

    @staticmethod
    def _validate_knowledge_extraction(
        document: ParsedDocument,
        extraction: ChunkKnowledge,
        chunks_by_id: dict[str, TextChunk],
        seen_extractions: set[str],
    ) -> None:
        if extraction.document_id != document.document_id:
            raise ValueError(
                f"Knowledge extraction {extraction.chunk_id} belongs to "
                f"{extraction.document_id}, not {document.document_id}"
            )
        if extraction.chunk_id not in chunks_by_id:
            raise KeyError(
                f"Knowledge source chunk not found: {extraction.chunk_id}"
            )
        if extraction.chunk_id in seen_extractions:
            raise ValueError(
                f"Duplicate knowledge extraction: {extraction.chunk_id}"
            )
        entity_names = {
            LightRAGIndexer._entity_key(entity.entity_name)
            for entity in extraction.entities
        }
        for relationship in extraction.relationships:
            if (
                LightRAGIndexer._entity_key(relationship.source_entity)
                not in entity_names
                or LightRAGIndexer._entity_key(relationship.target_entity)
                not in entity_names
            ):
                raise ValueError(
                    "Knowledge relationship endpoint is not declared in its "
                    f"chunk: {relationship.source_entity!r} -> "
                    f"{relationship.target_entity!r}"
                )
        seen_extractions.add(extraction.chunk_id)

    @staticmethod
    def _entity_key(value: str) -> str:
        return " ".join(value.split()).casefold()

    @staticmethod
    def _merge_descriptions(descriptions: list[str], limit: int = 2000) -> str:
        merged = " ".join(descriptions)
        if len(merged) <= limit:
            return merged
        return merged[: limit - 3].rstrip() + "..."

    @staticmethod
    def _validate_analysis(
        document: ParsedDocument,
        analysis: ModalAnalysis,
        items_by_id: dict[str, Any],
        seen_item_ids: set[str],
    ) -> None:
        if analysis.document_id != document.document_id:
            raise ValueError(
                f"Analysis {analysis.item_id} belongs to "
                f"{analysis.document_id}, not {document.document_id}"
            )
        if analysis.item_id not in items_by_id:
            raise KeyError(
                f"Analysis source item not found: {analysis.item_id}"
            )
        if analysis.item_id in seen_item_ids:
            raise ValueError(
                f"Duplicate analysis for item: {analysis.item_id}"
            )
        item = items_by_id[analysis.item_id]
        if item.type != analysis.content_type:
            raise ValueError(
                f"Analysis type {analysis.content_type.value} does not match "
                f"item type {item.type.value}: {analysis.item_id}"
            )
        if not analysis.chunk_text.strip():
            raise ValueError(
                f"Analysis chunk is empty: {analysis.item_id}"
            )
        seen_item_ids.add(analysis.item_id)

    @staticmethod
    def _validate_text_chunk(
        document: ParsedDocument,
        chunk: TextChunk,
        items_by_id: dict[str, Any],
        seen_chunk_ids: set[str],
    ) -> None:
        if chunk.document_id != document.document_id:
            raise ValueError(
                f"Text chunk {chunk.chunk_id} belongs to "
                f"{chunk.document_id}, not {document.document_id}"
            )
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate text chunk: {chunk.chunk_id}")
        if not chunk.text.strip():
            raise ValueError(f"Text chunk is empty: {chunk.chunk_id}")
        if chunk.token_count <= 0:
            raise ValueError(f"Text chunk token count is invalid: {chunk.chunk_id}")
        missing_item_ids = [
            item_id
            for item_id in chunk.source_item_ids
            if item_id not in items_by_id
        ]
        if missing_item_ids:
            raise KeyError(
                f"Text chunk source items not found: {missing_item_ids}"
            )
        seen_chunk_ids.add(chunk.chunk_id)

    @staticmethod
    def _unique_entity_name(
        analysis: ModalAnalysis,
        existing_names: set[str],
    ) -> str:
        base_name = analysis.entity_info.entity_name.strip()
        if not base_name:
            base_name = analysis.item_id
        typed_name = f"{base_name} ({analysis.content_type.value})"
        if typed_name not in existing_names:
            return typed_name
        return f"{typed_name} [{analysis.item_id[-8:]}]"
