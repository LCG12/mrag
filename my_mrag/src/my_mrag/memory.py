from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from my_mrag.schemas import (
    AnswerResult,
    AnswerSource,
    ConversationSession,
    ConversationTurn,
)
from my_mrag.storage import JsonConversationStore
from my_mrag.utils import stable_id


@dataclass(frozen=True)
class MemoryConfig:
    context_turns: int = 4
    max_answer_characters: int = 1600
    max_context_characters: int = 8000
    max_sources_per_turn: int = 5

    def __post_init__(self) -> None:
        if self.context_turns <= 0:
            raise ValueError("context_turns must be positive")
        if self.max_answer_characters <= 0:
            raise ValueError("max_answer_characters must be positive")
        if self.max_context_characters <= 0:
            raise ValueError("max_context_characters must be positive")
        if self.max_sources_per_turn < 0:
            raise ValueError("max_sources_per_turn cannot be negative")


class ConversationMemory:
    """Build bounded chat context and persist completed grounded turns."""

    def __init__(
        self,
        store: JsonConversationStore,
        config: MemoryConfig | None = None,
    ):
        self.store = store
        self.config = config or MemoryConfig()

    def context(self, session_id: str) -> str:
        if not self.store.exists(session_id):
            return ""
        session = self.store.load(session_id)
        recent_turns = session.turns[-self.config.context_turns :]
        blocks: list[str] = []
        used = 0
        for turn in reversed(recent_turns):
            block = self._format_turn(turn)
            separator_cost = 2 if blocks else 0
            available = (
                self.config.max_context_characters
                - used
                - separator_cost
            )
            if available <= 0:
                break
            if len(block) > available:
                if not blocks:
                    blocks.append(self._truncate(block, available))
                break
            blocks.append(block)
            used += len(block) + separator_cost
        blocks.reverse()
        return "\n\n".join(blocks)

    def remember(
        self,
        session_id: str,
        query: str,
        answer: AnswerResult,
        *,
        document_id: str | None = None,
    ) -> tuple[ConversationSession, Path]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Memory query cannot be empty")
        session = self.store.load_or_create(session_id)
        cited_ids = set(answer.cited_source_ids)
        cited_sources = tuple(
            source
            for source in answer.sources
            if source.citation_id in cited_ids
        )
        turn = ConversationTurn(
            turn_id=stable_id(
                "turn",
                session.session_id,
                len(session.turns) + 1,
                normalized_query,
                answer.answer,
            ),
            query=normalized_query,
            answer=answer.answer,
            document_id=document_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            cited_sources=cited_sources,
        )
        return self.store.append(session.session_id, turn)

    def _format_turn(self, turn: ConversationTurn) -> str:
        truncated_answer = self._truncate(
            turn.answer,
            self.config.max_answer_characters,
        )
        answer = self._escape_context(truncated_answer)
        answer = re.sub(
            r"\[S(\d+)\]",
            r"[prior citation S\1]",
            answer,
        )
        source_lines = [
            self._format_source(source)
            for source in turn.cited_sources[
                : self.config.max_sources_per_turn
            ]
        ]
        sources = "\n".join(source_lines) or "- None"
        return (
            f"Previous turn {turn.turn_id}\n"
            f"Document ID: {turn.document_id or 'not limited'}\n"
            f"User: {self._escape_context(turn.query)}\n"
            f"Assistant: {answer}\n"
            f"Previously cited sources:\n{sources}"
        )

    @staticmethod
    def _format_source(source: AnswerSource) -> str:
        name = Path(source.file_path).name or source.document_id
        if source.page_start is None:
            pages = "unknown"
        elif source.page_end in (None, source.page_start):
            pages = str(source.page_start)
        else:
            pages = f"{source.page_start}-{source.page_end}"
        section = " > ".join(source.section_path) or "None"
        formatted = (
            f"- {name}; pages {pages}; section {section}; "
            f"modality {source.content_type}"
        )
        return ConversationMemory._escape_context(formatted)

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        normalized = value.strip()
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return normalized[:limit]
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _escape_context(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
