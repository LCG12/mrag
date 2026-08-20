from __future__ import annotations

from pathlib import Path

import pytest

from my_mrag.memory import ConversationMemory, MemoryConfig
from my_mrag.schemas import AnswerResult, AnswerSource
from my_mrag.storage import JsonConversationStore


def _source(citation_id: str, page: int) -> AnswerSource:
    return AnswerSource(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        source_id=f"source-{citation_id}",
        document_id="doc-paper",
        file_path="papers/paper.pdf",
        page_start=page,
        page_end=page,
        section_path=("Results",),
        content_type="text",
        score=0.8,
    )


def _answer(query: str, text: str) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer=text,
        sources=(_source("S1", 2), _source("S2", 5)),
        cited_source_ids=("S2",),
        model_name="answer-model",
    )


def test_memory_persists_only_cited_sources_and_round_trips(
    tmp_path: Path,
) -> None:
    store = JsonConversationStore(tmp_path / "memory")
    memory = ConversationMemory(store)

    session, stored_path = memory.remember(
        "research_1",
        "What is CPS?",
        _answer("What is CPS?", "CPS combines stability and accuracy [S2]."),
        document_id="doc-paper",
    )

    assert stored_path.is_file()
    assert len(session.turns) == 1
    loaded = store.load("research_1")
    assert loaded == session
    assert [source.citation_id for source in loaded.turns[0].cited_sources] == [
        "S2"
    ]
    context = memory.context("research_1")
    assert "What is CPS?" in context
    assert "CPS combines stability" in context
    assert "[prior citation S2]" in context
    assert "[S2]" not in context
    assert "pages 5" in context
    assert "pages 2" not in context


def test_memory_context_keeps_recent_turns_within_budget(
    tmp_path: Path,
) -> None:
    store = JsonConversationStore(tmp_path / "memory")
    memory = ConversationMemory(
        store,
        MemoryConfig(
            context_turns=1,
            max_answer_characters=20,
            max_context_characters=500,
        ),
    )
    memory.remember(
        "research-2",
        "First question",
        _answer("First question", "First answer"),
    )
    memory.remember(
        "research-2",
        "Second question",
        _answer("Second question", "A very long second answer for truncation"),
    )

    context = memory.context("research-2")

    assert "First question" not in context
    assert "Second question" in context
    assert "A very long secon..." in context


def test_conversation_store_rejects_unsafe_session_id(tmp_path: Path) -> None:
    store = JsonConversationStore(tmp_path / "memory")

    with pytest.raises(ValueError, match="session_id"):
        store.load_or_create("../outside")
