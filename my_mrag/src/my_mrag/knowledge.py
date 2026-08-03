from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from my_mrag.models import AnalysisModel
from my_mrag.prompts import (
    KNOWLEDGE_EXTRACTION_PROMPT,
    KNOWLEDGE_EXTRACTION_SYSTEM,
)
from my_mrag.schemas import (
    AnalysisRequest,
    ChunkKnowledge,
    ContentType,
    KnowledgeEntity,
    KnowledgeRelationship,
    TextChunk,
)


@dataclass(frozen=True)
class KnowledgeExtractionConfig:
    max_entities: int = 12
    max_relationships: int = 16
    concurrency: int = 2
    retries: int = 1

    def __post_init__(self) -> None:
        if self.max_entities <= 0:
            raise ValueError("max_entities must be positive")
        if self.max_relationships < 0:
            raise ValueError("max_relationships cannot be negative")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")


class KnowledgeExtractionPipeline:
    """Extract an evidence-grounded graph from retrieval-ready text chunks."""

    def __init__(
        self,
        model: AnalysisModel,
        config: KnowledgeExtractionConfig | None = None,
    ):
        self.model = model
        self.config = config or KnowledgeExtractionConfig()

    def prepare(self, chunk: TextChunk) -> AnalysisRequest:
        section_path = " > ".join(chunk.section_path) or "None"
        return AnalysisRequest(
            item_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content_type=ContentType.TEXT,
            system_prompt=KNOWLEDGE_EXTRACTION_SYSTEM,
            prompt=KNOWLEDGE_EXTRACTION_PROMPT.format(
                chunk_id=chunk.chunk_id,
                page_start=chunk.page_start + 1,
                page_end=chunk.page_end + 1,
                section_path=section_path,
                text=chunk.text,
                max_entities=self.config.max_entities,
                max_relationships=self.config.max_relationships,
            ),
            context=section_path,
        )

    async def extract(self, chunks: list[TextChunk]) -> list[ChunkKnowledge]:
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def run(chunk: TextChunk) -> ChunkKnowledge:
            async with semaphore:
                return await self.extract_chunk(chunk)

        return list(await asyncio.gather(*(run(chunk) for chunk in chunks)))

    async def extract_chunk(self, chunk: TextChunk) -> ChunkKnowledge:
        request = self.prepare(chunk)
        last_error: ValueError | None = None
        for _ in range(self.config.retries + 1):
            response = await self.model.complete(request)
            try:
                return self._parse_response(response, chunk)
            except ValueError as error:
                last_error = error
        raise RuntimeError(
            f"Knowledge extraction failed for {chunk.chunk_id}: {last_error}"
        ) from last_error

    def _parse_response(
        self,
        response: str,
        chunk: TextChunk,
    ) -> ChunkKnowledge:
        cleaned = self._strip_thinking_tags(response)
        payload = self._parse_json_object(cleaned)
        if payload is None:
            raise ValueError("model response does not contain a JSON object")

        raw_entities = payload.get("entities")
        raw_relationships = payload.get("relationships")
        if not isinstance(raw_entities, list):
            raise ValueError("entities must be a JSON array")
        if not isinstance(raw_relationships, list):
            raise ValueError("relationships must be a JSON array")
        if len(raw_entities) > self.config.max_entities:
            raise ValueError(
                f"model returned more than {self.config.max_entities} entities"
            )
        if len(raw_relationships) > self.config.max_relationships:
            raise ValueError(
                "model returned more than "
                f"{self.config.max_relationships} relationships"
            )

        entities = self._parse_entities(raw_entities)
        canonical_names = {
            self._name_key(entity.entity_name): entity.entity_name
            for entity in entities
        }
        relationships = self._parse_relationships(
            raw_relationships,
            canonical_names,
        )
        return ChunkKnowledge(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            entities=tuple(entities),
            relationships=tuple(relationships),
            model_name=self.model.model_name,
        )

    def _parse_entities(self, values: list[Any]) -> list[KnowledgeEntity]:
        entities: dict[str, KnowledgeEntity] = {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("each entity must be a JSON object")
            try:
                entity = KnowledgeEntity.from_dict(value)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid entity: {error}") from error
            key = self._name_key(entity.entity_name)
            existing = entities.get(key)
            if existing is None:
                entities[key] = entity
            elif entity.description not in existing.description:
                entities[key] = KnowledgeEntity(
                    entity_name=existing.entity_name,
                    entity_type=existing.entity_type,
                    description=(
                        f"{existing.description} {entity.description}"
                    ),
                )
        return list(entities.values())

    def _parse_relationships(
        self,
        values: list[Any],
        canonical_names: dict[str, str],
    ) -> list[KnowledgeRelationship]:
        relationships: dict[tuple[str, str], KnowledgeRelationship] = {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("each relationship must be a JSON object")
            try:
                relationship = KnowledgeRelationship.from_dict(value)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid relationship: {error}") from error

            source = canonical_names.get(
                self._name_key(relationship.source_entity)
            )
            target = canonical_names.get(
                self._name_key(relationship.target_entity)
            )
            if source is None or target is None:
                raise ValueError(
                    "relationship endpoints must reference extracted entities: "
                    f"{relationship.source_entity!r} -> "
                    f"{relationship.target_entity!r}"
                )
            if source == target:
                continue

            normalized = KnowledgeRelationship(
                source_entity=source,
                target_entity=target,
                description=relationship.description,
                keywords=relationship.keywords or ("related_to",),
                weight=min(max(relationship.weight, 1.0), 10.0),
            )
            key = tuple(sorted((self._name_key(source), self._name_key(target))))
            existing = relationships.get(key)
            if existing is None:
                relationships[key] = normalized
            else:
                relationships[key] = KnowledgeRelationship(
                    source_entity=existing.source_entity,
                    target_entity=existing.target_entity,
                    description=self._merge_text(
                        existing.description,
                        normalized.description,
                    ),
                    keywords=tuple(
                        dict.fromkeys(
                            [*existing.keywords, *normalized.keywords]
                        )
                    ),
                    weight=max(existing.weight, normalized.weight),
                )
        return list(relationships.values())

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

    @staticmethod
    def _name_key(value: str) -> str:
        return " ".join(value.split()).casefold()

    @staticmethod
    def _merge_text(first: str, second: str) -> str:
        return first if second in first else f"{first} {second}"
