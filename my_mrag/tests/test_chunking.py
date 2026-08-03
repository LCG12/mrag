from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import pytest

from my_mrag.chunking import TextChunkConfig, TextChunker
from my_mrag.schemas import ContentItem, ContentType, ParsedDocument
from my_mrag.storage import JsonTextChunkStore


class WordTokenizer:
    name = "test-word-tokenizer"

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\S+\s*", text)

    def decode(self, tokens: Sequence[Any]) -> str:
        return "".join(str(token) for token in tokens)


def _text_item(
    item_id: str,
    text: str,
    order_idx: int,
    page_idx: int = 0,
) -> ContentItem:
    return ContentItem(
        item_id=item_id,
        document_id="doc-1",
        type=ContentType.TEXT,
        page_idx=page_idx,
        order_idx=order_idx,
        text=text,
    )


def _document(items: list[ContentItem]) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_path=str(Path("papers") / "paper.pdf"),
        parser="test",
        page_count=3,
        items=items,
    )


def test_chunker_respects_sections_token_limits_and_overlap() -> None:
    document = _document(
        [
            _text_item("heading-1", "1 INTRODUCTION", 0),
            _text_item(
                "intro-1",
                "alpha beta gamma delta epsilon zeta",
                1,
            ),
            _text_item(
                "intro-2",
                "eta theta iota kappa lambda mu",
                2,
                page_idx=1,
            ),
            _text_item("heading-2", "2 METHODS", 3, page_idx=1),
            _text_item(
                "methods-1",
                "planner retrieves evidence before executor acts",
                4,
                page_idx=2,
            ),
        ]
    )
    chunker = TextChunker(
        TextChunkConfig(
            target_tokens=8,
            max_tokens=10,
            overlap_tokens=2,
        ),
        tokenizer=WordTokenizer(),
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert chunks[0].section_path == ("1 INTRODUCTION",)
    assert chunks[1].section_path == ("1 INTRODUCTION",)
    assert chunks[2].section_path == ("2 METHODS",)
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert chunks[0].text.endswith("epsilon zeta")
    assert chunks[1].text.startswith("epsilon zeta")
    assert chunks[1].page_start == 0
    assert chunks[1].page_end == 1
    assert chunks[2].index_text().startswith("Section: 2 METHODS")


def test_chunker_splits_one_oversized_text_item() -> None:
    long_text = " ".join(f"token-{index}" for index in range(25))
    document = _document([_text_item("long", long_text, 0)])
    chunker = TextChunker(
        TextChunkConfig(
            target_tokens=8,
            max_tokens=10,
            overlap_tokens=2,
        ),
        tokenizer=WordTokenizer(),
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert all(chunk.source_item_ids == ("long",) for chunk in chunks)


def test_chunker_does_not_treat_acronyms_or_equations_as_headings() -> None:
    document = _document(
        [
            _text_item("heading", "2 METHODS", 0),
            _text_item("acronym", "VLM/LLM", 1),
            _text_item("equation", "T = emb(s) : s in V", 2),
            _text_item("body", "Evidence is embedded for retrieval.", 3),
        ]
    )

    chunks = TextChunker(tokenizer=WordTokenizer()).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].section_path == ("2 METHODS",)
    assert "VLM/LLM" in chunks[0].text
    assert "T = emb(s)" in chunks[0].text


def test_chunk_store_round_trips(tmp_path: Path) -> None:
    document = _document(
        [_text_item("text-1", "retrieval ready scientific evidence", 0)]
    )
    chunks = TextChunker(tokenizer=WordTokenizer()).chunk(document)
    store = JsonTextChunkStore(tmp_path / "chunks")

    stored_path = store.save(document.document_id, chunks)
    loaded = store.load(document.document_id)

    assert stored_path.is_file()
    assert [chunk.to_dict() for chunk in loaded] == [
        chunk.to_dict() for chunk in chunks
    ]


def test_chunk_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="smaller than target_tokens"):
        TextChunkConfig(
            target_tokens=100,
            max_tokens=200,
            overlap_tokens=100,
        )
