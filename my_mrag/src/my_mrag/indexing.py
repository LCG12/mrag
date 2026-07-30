from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from my_mrag.schemas import ModalAnalysis, ParsedDocument


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
    ) -> dict[str, list[dict[str, Any]]]:
        if not analyses:
            return {"chunks": [], "entities": [], "relationships": []}

        items_by_id = {item.item_id: item for item in document.items}
        seen_item_ids: set[str] = set()
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

        # LightRAG requires every entity source_id to map to a custom chunk.
        first_source_id = analyses[0].item_id
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
    ) -> IndexingReport:
        payload = self.build_payload(document, analyses)
        if payload["chunks"]:
            await self.store.ainsert_custom_kg(
                payload,
                full_doc_id=document.document_id,
            )
        return IndexingReport(
            document_id=document.document_id,
            chunk_count=len(payload["chunks"]),
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
