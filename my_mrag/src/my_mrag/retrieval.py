from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from my_mrag.config import Settings
from my_mrag.schemas import (
    ContentItem,
    ParsedDocument,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    TextChunk,
)
from my_mrag.storage import JsonDocumentStore, JsonTextChunkStore


_SOURCE_ID_SEPARATOR = "<SEP>"


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_top_k: int = 8
    entity_top_k: int = 5
    relationship_top_k: int = 5
    final_top_k: int = 8
    candidate_multiplier: int = 3
    graph_depth: int = 1
    graph_max_neighbors: int = 20
    entity_source_weight: float = 0.9
    relationship_source_weight: float = 0.85
    graph_decay: float = 0.8

    def __post_init__(self) -> None:
        integer_values = {
            "chunk_top_k": self.chunk_top_k,
            "entity_top_k": self.entity_top_k,
            "relationship_top_k": self.relationship_top_k,
            "final_top_k": self.final_top_k,
            "candidate_multiplier": self.candidate_multiplier,
            "graph_depth": self.graph_depth,
            "graph_max_neighbors": self.graph_max_neighbors,
        }
        if any(value < 0 for value in integer_values.values()):
            raise ValueError("Retrieval integer settings cannot be negative")
        if self.final_top_k == 0:
            raise ValueError("final_top_k must be positive")
        if self.candidate_multiplier == 0:
            raise ValueError("candidate_multiplier must be positive")
        for name in (
            "entity_source_weight",
            "relationship_source_weight",
            "graph_decay",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


class RetrievalPipeline:
    """Retrieve chunks and expand vector-matched graph evidence."""

    def __init__(
        self,
        rag: Any,
        settings: Settings,
        config: RetrievalConfig | None = None,
    ):
        self.rag = rag
        self.settings = settings
        self.config = config or RetrievalConfig()
        self.document_store = JsonDocumentStore(settings.parsed_dir)
        self.chunk_store = JsonTextChunkStore(settings.chunks_dir)
        self._documents: dict[str, ParsedDocument | None] = {}
        self._text_chunks: dict[str, dict[str, TextChunk]] = {}
        self._light_chunks: dict[str, dict[str, Any] | None] = {}
        self._nodes: dict[str, dict[str, Any] | None] = {}
        self._edges: dict[tuple[str, str], dict[str, Any] | None] = {}

    async def retrieve(
        self,
        query: str,
        *,
        document_id: str | None = None,
    ) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Retrieval query cannot be empty")

        query_embedding = await self._embed_query(normalized_query)
        chunk_hits, entity_hits, relationship_hits = await asyncio.gather(
            self._vector_query(
                self.rag.chunks_vdb,
                normalized_query,
                self.config.chunk_top_k,
                query_embedding,
            ),
            self._vector_query(
                self.rag.entities_vdb,
                normalized_query,
                self.config.entity_top_k,
                query_embedding,
            ),
            self._vector_query(
                self.rag.relationships_vdb,
                normalized_query,
                self.config.relationship_top_k,
                query_embedding,
            ),
        )

        chunk_candidates: dict[str, dict[str, Any]] = {}
        entity_candidates: dict[str, dict[str, Any]] = {}
        relationship_candidates: dict[tuple[str, str], dict[str, Any]] = {}

        direct_chunk_records = await self._records_for_hits(chunk_hits)
        for hit, record in direct_chunk_records:
            if not self._record_matches_document(record, document_id):
                continue
            self._add_chunk_candidate(
                chunk_candidates,
                record,
                self._score(hit),
                "chunk_vector",
            )

        graph_seeds: dict[str, float] = {}
        for hit in entity_hits:
            entity_name = str(hit.get("entity_name") or "").strip()
            if not entity_name:
                continue
            node = await self._get_node(entity_name)
            source_ids = self._source_ids(
                (node or {}).get("source_id") or hit.get("source_id")
            )
            source_records = await self._get_light_chunks(source_ids)
            matching_records = self._matching_records(
                source_records,
                document_id,
            )
            if document_id and not matching_records and not entity_name.endswith(
                document_id
            ):
                continue

            score = self._score(hit)
            self._add_entity_candidate(
                entity_candidates,
                entity_name,
                node or hit,
                score,
                source_ids,
                "entity_vector",
            )
            graph_seeds[entity_name] = max(graph_seeds.get(entity_name, 0), score)
            if not self._is_document_entity(entity_name, node or hit):
                for record in matching_records:
                    self._add_chunk_candidate(
                        chunk_candidates,
                        record,
                        score * self.config.entity_source_weight,
                        "entity_source",
                    )

        for hit in relationship_hits:
            source_entity = str(hit.get("src_id") or "").strip()
            target_entity = str(hit.get("tgt_id") or "").strip()
            if not source_entity or not target_entity:
                continue
            edge = await self._get_edge(source_entity, target_entity)
            relation_data = edge or hit
            source_ids = self._source_ids(
                relation_data.get("source_id") or hit.get("source_id")
            )
            source_records = await self._get_light_chunks(source_ids)
            matching_records = self._matching_records(
                source_records,
                document_id,
            )
            if document_id and not matching_records:
                continue

            score = self._score(hit)
            self._add_relationship_candidate(
                relationship_candidates,
                source_entity,
                target_entity,
                relation_data,
                score,
                source_ids,
                "relationship_vector",
            )
            graph_seeds[source_entity] = max(
                graph_seeds.get(source_entity, 0),
                score,
            )
            graph_seeds[target_entity] = max(
                graph_seeds.get(target_entity, 0),
                score,
            )
            for record in matching_records:
                self._add_chunk_candidate(
                    chunk_candidates,
                    record,
                    score * self.config.relationship_source_weight,
                    "relationship_source",
                )

        await self._expand_graph(
            graph_seeds=graph_seeds,
            chunk_candidates=chunk_candidates,
            entity_candidates=entity_candidates,
            relationship_candidates=relationship_candidates,
            document_id=document_id,
        )

        chunks = [
            await self._resolve_chunk(candidate)
            for candidate in sorted(
                chunk_candidates.values(),
                key=lambda value: value["score"],
                reverse=True,
            )[: self.config.final_top_k]
        ]
        entities = tuple(
            self._build_entity(candidate)
            for candidate in sorted(
                entity_candidates.values(),
                key=lambda value: value["score"],
                reverse=True,
            )
        )
        relationships = tuple(
            self._build_relationship(candidate)
            for candidate in sorted(
                relationship_candidates.values(),
                key=lambda value: value["score"],
                reverse=True,
            )
        )
        return RetrievalResult(
            query=normalized_query,
            chunks=tuple(chunks),
            entities=entities,
            relationships=relationships,
        )

    async def _expand_graph(
        self,
        graph_seeds: dict[str, float],
        chunk_candidates: dict[str, dict[str, Any]],
        entity_candidates: dict[str, dict[str, Any]],
        relationship_candidates: dict[tuple[str, str], dict[str, Any]],
        document_id: str | None,
    ) -> None:
        frontier = dict(graph_seeds)
        visited: set[str] = set()
        for depth in range(self.config.graph_depth):
            next_frontier: dict[str, float] = {}
            for entity_name, seed_score in frontier.items():
                if entity_name in visited:
                    continue
                visited.add(entity_name)
                edges = (
                    await self.rag.chunk_entity_relation_graph.get_node_edges(
                        entity_name
                    )
                    or []
                )
                for source_entity, target_entity in edges[
                    : self.config.graph_max_neighbors
                ]:
                    edge = await self._get_edge(source_entity, target_entity)
                    if edge is None:
                        continue
                    source_ids = self._source_ids(edge.get("source_id"))
                    source_records = await self._get_light_chunks(source_ids)
                    matching_records = self._matching_records(
                        source_records,
                        document_id,
                    )
                    if document_id and not matching_records:
                        continue

                    graph_score = seed_score * (
                        self.config.graph_decay ** (depth + 1)
                    )
                    self._add_relationship_candidate(
                        relationship_candidates,
                        source_entity,
                        target_entity,
                        edge,
                        graph_score,
                        source_ids,
                        "graph_expand",
                    )
                    for record in matching_records:
                        self._add_chunk_candidate(
                            chunk_candidates,
                            record,
                            graph_score * self.config.relationship_source_weight,
                            "graph_source",
                        )

                    neighbor = (
                        target_entity
                        if source_entity == entity_name
                        else source_entity
                    )
                    neighbor_node = await self._get_node(neighbor)
                    if neighbor_node is not None:
                        neighbor_source_ids = self._source_ids(
                            neighbor_node.get("source_id")
                        )
                        self._add_entity_candidate(
                            entity_candidates,
                            neighbor,
                            neighbor_node,
                            graph_score,
                            neighbor_source_ids,
                            "graph_expand",
                        )
                        if not self._is_document_entity(
                            neighbor,
                            neighbor_node,
                        ):
                            neighbor_records = await self._get_light_chunks(
                                neighbor_source_ids
                            )
                            for record in self._matching_records(
                                neighbor_records,
                                document_id,
                            ):
                                self._add_chunk_candidate(
                                    chunk_candidates,
                                    record,
                                    graph_score * self.config.entity_source_weight,
                                    "graph_source",
                                )
                    next_frontier[neighbor] = max(
                        next_frontier.get(neighbor, 0),
                        graph_score,
                    )
            frontier = next_frontier

    async def _embed_query(self, query: str) -> list[float]:
        embeddings = await self.rag.embedding_func(
            [query],
            context="query",
        )
        vector = embeddings[0]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    async def _vector_query(
        self,
        storage: Any,
        query: str,
        top_k: int,
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        if top_k == 0:
            return []
        candidate_k = max(top_k, top_k * self.config.candidate_multiplier)
        return await storage.query(
            query,
            top_k=candidate_k,
            query_embedding=query_embedding,
        )

    async def _records_for_hits(
        self,
        hits: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        records = await self._get_light_chunks(
            [str(hit.get("id") or "") for hit in hits]
        )
        records_by_id = {
            self._light_chunk_id(record): record
            for record in records
        }
        return [
            (hit, records_by_id[hit_id])
            for hit in hits
            if (hit_id := str(hit.get("id") or "")) in records_by_id
        ]

    async def _get_light_chunks(
        self,
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        unique_ids = list(dict.fromkeys(chunk_id for chunk_id in chunk_ids if chunk_id))
        missing_ids = [
            chunk_id
            for chunk_id in unique_ids
            if chunk_id not in self._light_chunks
        ]
        if missing_ids:
            loaded = await asyncio.gather(
                *(
                    self.rag.text_chunks.get_by_id(chunk_id)
                    for chunk_id in missing_ids
                )
            )
            for chunk_id, record in zip(missing_ids, loaded):
                self._light_chunks[chunk_id] = record
        return [
            record
            for chunk_id in unique_ids
            if (record := self._light_chunks.get(chunk_id)) is not None
        ]

    async def _get_node(self, entity_name: str) -> dict[str, Any] | None:
        if entity_name not in self._nodes:
            self._nodes[entity_name] = (
                await self.rag.chunk_entity_relation_graph.get_node(entity_name)
            )
        return self._nodes[entity_name]

    async def _get_edge(
        self,
        source_entity: str,
        target_entity: str,
    ) -> dict[str, Any] | None:
        key = self._relationship_key(source_entity, target_entity)
        if key not in self._edges:
            self._edges[key] = (
                await self.rag.chunk_entity_relation_graph.get_edge(
                    source_entity,
                    target_entity,
                )
            )
        return self._edges[key]

    async def _resolve_chunk(
        self,
        candidate: dict[str, Any],
    ) -> RetrievedChunk:
        record = candidate["record"]
        document_id = str(record.get("full_doc_id") or "")
        source_id = str(record.get("source_id") or "")
        document, text_chunks = self._load_local_sources(document_id)
        text_chunk = text_chunks.get(source_id)
        item = self._find_item(document, source_id)

        page_start: int | None = None
        page_end: int | None = None
        section_path: tuple[str, ...] = ()
        content_type = "text"
        asset_path: str | None = None
        captions: tuple[str, ...] = ()
        metadata: dict[str, Any] = {}
        if text_chunk is not None:
            page_start = text_chunk.page_start + 1
            page_end = text_chunk.page_end + 1
            section_path = text_chunk.section_path
            metadata = {
                "local_chunk_id": text_chunk.chunk_id,
                "chunk_index": text_chunk.chunk_index,
                "token_count": text_chunk.token_count,
                "source_item_ids": list(text_chunk.source_item_ids),
            }
        elif item is not None:
            page_start = item.page_idx + 1
            page_end = item.page_idx + 1
            content_type = item.type.value
            asset_path = item.asset_path
            captions = tuple(item.captions)
            metadata = {"item_id": item.item_id}

        return RetrievedChunk(
            chunk_id=self._light_chunk_id(record),
            score=round(float(candidate["score"]), 6),
            content=str(record.get("content") or ""),
            document_id=document_id,
            source_id=source_id,
            file_path=str(
                record.get("file_path")
                or (document.source_path if document else "")
            ),
            chunk_order_index=int(record.get("chunk_order_index") or 0),
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            content_type=content_type,
            asset_path=asset_path,
            captions=captions,
            channels=tuple(sorted(candidate["channels"])),
            metadata=metadata,
        )

    def _load_local_sources(
        self,
        document_id: str,
    ) -> tuple[ParsedDocument | None, dict[str, TextChunk]]:
        if document_id not in self._documents:
            self._documents[document_id] = (
                self.document_store.load(document_id)
                if document_id and self.document_store.exists(document_id)
                else None
            )
        if document_id not in self._text_chunks:
            chunks = (
                self.chunk_store.load(document_id)
                if document_id and self.chunk_store.exists(document_id)
                else []
            )
            self._text_chunks[document_id] = {
                chunk.chunk_id: chunk for chunk in chunks
            }
        return self._documents[document_id], self._text_chunks[document_id]

    @staticmethod
    def _find_item(
        document: ParsedDocument | None,
        item_id: str,
    ) -> ContentItem | None:
        if document is None:
            return None
        return next(
            (item for item in document.items if item.item_id == item_id),
            None,
        )

    @staticmethod
    def _add_chunk_candidate(
        candidates: dict[str, dict[str, Any]],
        record: dict[str, Any],
        score: float,
        channel: str,
    ) -> None:
        chunk_id = RetrievalPipeline._light_chunk_id(record)
        if not chunk_id:
            return
        candidate = candidates.setdefault(
            chunk_id,
            {
                "record": record,
                "score": float(score),
                "channels": set(),
            },
        )
        candidate["score"] = max(candidate["score"], float(score))
        candidate["channels"].add(channel)

    @staticmethod
    def _add_entity_candidate(
        candidates: dict[str, dict[str, Any]],
        entity_name: str,
        data: dict[str, Any],
        score: float,
        source_ids: list[str],
        channel: str,
    ) -> None:
        candidate = candidates.setdefault(
            entity_name,
            {
                "entity_name": entity_name,
                "entity_type": str(data.get("entity_type") or "unknown"),
                "description": str(data.get("description") or data.get("content") or ""),
                "score": float(score),
                "source_ids": set(),
                "channels": set(),
            },
        )
        candidate["score"] = max(candidate["score"], float(score))
        candidate["source_ids"].update(source_ids)
        candidate["channels"].add(channel)

    @staticmethod
    def _add_relationship_candidate(
        candidates: dict[tuple[str, str], dict[str, Any]],
        source_entity: str,
        target_entity: str,
        data: dict[str, Any],
        score: float,
        source_ids: list[str],
        channel: str,
    ) -> None:
        key = RetrievalPipeline._relationship_key(source_entity, target_entity)
        candidate = candidates.setdefault(
            key,
            {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "description": str(data.get("description") or data.get("content") or ""),
                "keywords": str(data.get("keywords") or ""),
                "weight": RetrievalPipeline._float(data.get("weight"), 1.0),
                "score": float(score),
                "source_ids": set(),
                "channels": set(),
            },
        )
        candidate["score"] = max(candidate["score"], float(score))
        candidate["source_ids"].update(source_ids)
        candidate["channels"].add(channel)

    @staticmethod
    def _build_entity(candidate: dict[str, Any]) -> RetrievedEntity:
        return RetrievedEntity(
            entity_name=candidate["entity_name"],
            entity_type=candidate["entity_type"],
            description=candidate["description"],
            score=round(float(candidate["score"]), 6),
            source_chunk_ids=tuple(sorted(candidate["source_ids"])),
            channels=tuple(sorted(candidate["channels"])),
        )

    @staticmethod
    def _build_relationship(
        candidate: dict[str, Any],
    ) -> RetrievedRelationship:
        return RetrievedRelationship(
            source_entity=candidate["source_entity"],
            target_entity=candidate["target_entity"],
            description=candidate["description"],
            keywords=candidate["keywords"],
            score=round(float(candidate["score"]), 6),
            weight=float(candidate["weight"]),
            source_chunk_ids=tuple(sorted(candidate["source_ids"])),
            channels=tuple(sorted(candidate["channels"])),
        )

    @staticmethod
    def _source_ids(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [
            item.strip()
            for item in str(value).split(_SOURCE_ID_SEPARATOR)
            if item.strip()
        ]

    @staticmethod
    def _matching_records(
        records: list[dict[str, Any]],
        document_id: str | None,
    ) -> list[dict[str, Any]]:
        if document_id is None:
            return records
        return [
            record
            for record in records
            if RetrievalPipeline._record_matches_document(record, document_id)
        ]

    @staticmethod
    def _record_matches_document(
        record: dict[str, Any],
        document_id: str | None,
    ) -> bool:
        return document_id is None or record.get("full_doc_id") == document_id

    @staticmethod
    def _light_chunk_id(record: dict[str, Any]) -> str:
        return str(record.get("_id") or record.get("id") or "")

    @staticmethod
    def _relationship_key(
        source_entity: str,
        target_entity: str,
    ) -> tuple[str, str]:
        return tuple(sorted((source_entity, target_entity)))

    @staticmethod
    def _score(hit: dict[str, Any]) -> float:
        return RetrievalPipeline._float(hit.get("distance"), 0.0)

    @staticmethod
    def _is_document_entity(
        entity_name: str,
        data: dict[str, Any],
    ) -> bool:
        return (
            entity_name.startswith("DOCUMENT::")
            or str(data.get("entity_type") or "").lower() == "document"
        )

    @staticmethod
    def _float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
