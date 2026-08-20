from __future__ import annotations

import json
from pathlib import Path
import re

from my_mrag.schemas import (
    AnalysisRequest,
    ChunkKnowledge,
    ConversationSession,
    ConversationTurn,
    ModalAnalysis,
    ParsedDocument,
    TextChunk,
)


_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class JsonDocumentStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, document: ParsedDocument) -> Path:
        target = self.directory / f"{document.document_id}.json"
        temp_target = target.with_suffix(".json.tmp")
        with temp_target.open("w", encoding="utf-8") as stream:
            json.dump(document.to_dict(), stream, ensure_ascii=False, indent=2)
        temp_target.replace(target)
        return target

    def load(self, document_id: str) -> ParsedDocument:
        path = self.directory / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Parsed document not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            return ParsedDocument.from_dict(json.load(stream))

    def exists(self, document_id: str) -> bool:
        return (self.directory / f"{document_id}.json").is_file()


class JsonAnalysisStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_requests(
        self,
        document_id: str,
        requests: list[AnalysisRequest],
    ) -> Path:
        return self._save(
            self.directory / f"{document_id}.requests.json",
            {
                "document_id": document_id,
                "requests": [request.to_dict() for request in requests],
            },
        )

    def save_analyses(
        self,
        document_id: str,
        analyses: list[ModalAnalysis],
    ) -> Path:
        return self._save(
            self.directory / f"{document_id}.json",
            {
                "document_id": document_id,
                "analyses": [analysis.to_dict() for analysis in analyses],
            },
        )

    def load_analyses(self, document_id: str) -> list[ModalAnalysis]:
        path = self.directory / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Multimodal analysis not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return [
            ModalAnalysis.from_dict(item)
            for item in payload.get("analyses", [])
        ]

    def exists(self, document_id: str) -> bool:
        return (self.directory / f"{document_id}.json").is_file()

    @staticmethod
    def _save(path: Path, payload: dict[str, object]) -> Path:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        temp_path.replace(path)
        return path


class JsonTextChunkStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, chunks: list[TextChunk]) -> Path:
        target = self.directory / f"{document_id}.json"
        temp_target = target.with_suffix(".json.tmp")
        payload = {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        with temp_target.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        temp_target.replace(target)
        return target

    def load(self, document_id: str) -> list[TextChunk]:
        path = self.directory / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Text chunks not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("document_id") != document_id:
            raise ValueError(f"Text chunk document ID mismatch: {path}")
        return [
            TextChunk.from_dict(chunk)
            for chunk in payload.get("chunks", [])
        ]

    def exists(self, document_id: str) -> bool:
        return (self.directory / f"{document_id}.json").is_file()


class JsonKnowledgeStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        document_id: str,
        extractions: list[ChunkKnowledge],
    ) -> Path:
        target = self.directory / f"{document_id}.json"
        temp_target = target.with_suffix(".json.tmp")
        payload = {
            "document_id": document_id,
            "chunk_count": len(extractions),
            "entity_count": sum(
                len(extraction.entities) for extraction in extractions
            ),
            "relationship_count": sum(
                len(extraction.relationships) for extraction in extractions
            ),
            "extractions": [
                extraction.to_dict() for extraction in extractions
            ],
        }
        with temp_target.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        temp_target.replace(target)
        return target

    def merge_and_save(
        self,
        document_id: str,
        extractions: list[ChunkKnowledge],
    ) -> tuple[list[ChunkKnowledge], Path]:
        merged = {
            extraction.chunk_id: extraction
            for extraction in (
                self.load(document_id) if self.exists(document_id) else []
            )
        }
        merged.update(
            {extraction.chunk_id: extraction for extraction in extractions}
        )
        ordered = sorted(
            merged.values(),
            key=lambda extraction: extraction.chunk_index,
        )
        return ordered, self.save(document_id, ordered)

    def load(self, document_id: str) -> list[ChunkKnowledge]:
        path = self.directory / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge extraction not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("document_id") != document_id:
            raise ValueError(f"Knowledge extraction document ID mismatch: {path}")
        return [
            ChunkKnowledge.from_dict(extraction)
            for extraction in payload.get("extractions", [])
        ]

    def exists(self, document_id: str) -> bool:
        return (self.directory / f"{document_id}.json").is_file()


class JsonConversationStore:
    """Persist readable, append-only conversation sessions as JSON."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        normalized = self.validate_session_id(session_id)
        return self.directory / f"{normalized}.json"

    def save(self, session: ConversationSession) -> Path:
        target = self.path_for(session.session_id)
        temp_target = target.with_suffix(".json.tmp")
        with temp_target.open("w", encoding="utf-8") as stream:
            json.dump(
                session.to_dict(),
                stream,
                ensure_ascii=False,
                indent=2,
            )
        temp_target.replace(target)
        return target

    def load(self, session_id: str) -> ConversationSession:
        normalized = self.validate_session_id(session_id)
        path = self.path_for(normalized)
        if not path.is_file():
            raise FileNotFoundError(f"Conversation session not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            session = ConversationSession.from_dict(json.load(stream))
        if session.session_id != normalized:
            raise ValueError(f"Conversation session ID mismatch: {path}")
        return session

    def load_or_create(self, session_id: str) -> ConversationSession:
        normalized = self.validate_session_id(session_id)
        return (
            self.load(normalized)
            if self.exists(normalized)
            else ConversationSession(session_id=normalized)
        )

    def append(
        self,
        session_id: str,
        turn: ConversationTurn,
    ) -> tuple[ConversationSession, Path]:
        session = self.load_or_create(session_id)
        if any(existing.turn_id == turn.turn_id for existing in session.turns):
            raise ValueError(f"Duplicate conversation turn: {turn.turn_id}")
        updated = ConversationSession(
            session_id=session.session_id,
            turns=(*session.turns, turn),
        )
        return updated, self.save(updated)

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).is_file()

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        normalized = session_id.strip()
        if not _SESSION_ID_RE.fullmatch(normalized):
            raise ValueError(
                "session_id must contain 1-64 letters, digits, underscores, "
                "or hyphens and must start with a letter or digit"
            )
        return normalized
