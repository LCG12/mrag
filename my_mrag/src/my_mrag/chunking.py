from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from my_mrag.config import Settings
from my_mrag.schemas import ContentItem, ContentType, ParsedDocument, TextChunk
from my_mrag.utils import stable_id


_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<label>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))[.)]?\s+(?P<title>\S.*)$"
)
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d{1,4}$", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
_STANDALONE_HEADINGS = {
    "ABSTRACT",
    "ACKNOWLEDGMENTS",
    "APPENDIX",
    "CONCLUSION",
    "REFERENCES",
}


class ChunkTokenizer(Protocol):
    name: str

    def encode(self, text: str) -> Sequence[Any]:
        ...

    def decode(self, tokens: Sequence[Any]) -> str:
        ...


class ApproximateTokenizer:
    """Reversible approximation for environments without a model tokenizer."""

    name = "approximate-4-chars-or-1-cjk"

    def encode(self, text: str) -> list[str]:
        tokens: list[str] = []
        pending_space = ""
        index = 0
        while index < len(text):
            character = text[index]
            if character.isspace():
                start = index
                while index < len(text) and text[index].isspace():
                    index += 1
                pending_space += text[start:index]
                continue

            if character.isascii() and (character.isalnum() or character == "_"):
                start = index
                while index < len(text):
                    current = text[index]
                    if not (
                        current.isascii()
                        and (current.isalnum() or current == "_")
                    ):
                        break
                    index += 1
                word = text[start:index]
                pieces = [word[offset : offset + 4] for offset in range(0, len(word), 4)]
                pieces[0] = pending_space + pieces[0]
                pending_space = ""
                tokens.extend(pieces)
                continue

            tokens.append(pending_space + character)
            pending_space = ""
            index += 1

        if pending_space:
            if tokens:
                tokens[-1] += pending_space
            else:
                tokens.append(pending_space)
        return tokens

    def decode(self, tokens: Sequence[Any]) -> str:
        return "".join(str(token) for token in tokens)


class HuggingFaceTokenizer:
    def __init__(self, tokenizer: Any, model_path: Path):
        self.tokenizer = tokenizer
        self.name = f"huggingface:{model_path.name}"

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "HuggingFaceTokenizer":
        from transformers import AutoTokenizer

        path = Path(model_path).expanduser().resolve()
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
        )
        return cls(tokenizer, path)

    def encode(self, text: str) -> list[int]:
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def decode(self, tokens: Sequence[Any]) -> str:
        return str(
            self.tokenizer.decode(
                list(tokens),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )


def load_text_tokenizer(
    settings: Settings,
    *,
    force_approximate: bool = False,
) -> ChunkTokenizer:
    if force_approximate:
        return ApproximateTokenizer()

    configured_path = (
        os.getenv("TEXT_TOKENIZER_PATH")
        or os.getenv("EMBEDDING_MODEL_PATH")
    )
    if not configured_path:
        return ApproximateTokenizer()

    model_path = Path(configured_path).expanduser()
    if not model_path.is_absolute():
        model_path = settings.project_root / model_path
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Text tokenizer model not found: {model_path}")

    try:
        return HuggingFaceTokenizer.from_pretrained(model_path)
    except ImportError:
        return ApproximateTokenizer()


@dataclass(frozen=True)
class TextChunkConfig:
    target_tokens: int = 600
    max_tokens: int = 800
    overlap_tokens: int = 100

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be greater than or equal to target_tokens")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")


@dataclass(frozen=True)
class _TextUnit:
    text: str
    item_id: str
    page_idx: int
    order_idx: int


class TextChunker:
    """Build section-aware, overlapping retrieval chunks from parsed text."""

    def __init__(
        self,
        config: TextChunkConfig | None = None,
        tokenizer: ChunkTokenizer | None = None,
    ):
        self.config = config or TextChunkConfig()
        self.tokenizer = tokenizer or ApproximateTokenizer()

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        sections = self._collect_sections(document)
        chunks: list[TextChunk] = []
        for section_path, units in sections:
            chunks.extend(
                self._chunk_section(
                    document=document,
                    section_path=section_path,
                    units=units,
                    first_chunk_index=len(chunks),
                )
            )
        return chunks

    def _collect_sections(
        self,
        document: ParsedDocument,
    ) -> list[tuple[tuple[str, ...], list[_TextUnit]]]:
        boilerplate = self._find_repeated_boilerplate(document)
        sections: list[tuple[tuple[str, ...], list[_TextUnit]]] = []
        section_path: tuple[str, ...] = ()
        section_levels: tuple[int, ...] = ()
        units: list[_TextUnit] = []

        for item in sorted(document.items, key=lambda value: value.order_idx):
            if item.type != ContentType.TEXT:
                continue
            text = self._normalize_text(item.text)
            if not text or text in boilerplate or _PAGE_NUMBER_RE.fullmatch(text):
                continue

            metadata_headings = self._metadata_headings(item)
            if metadata_headings:
                for heading_level, heading_text in metadata_headings:
                    normalized_heading = self._normalize_text(heading_text)
                    if not text.casefold().startswith(
                        normalized_heading.casefold()
                    ):
                        break
                    if units:
                        sections.append((section_path, units))
                        units = []
                    section_path, section_levels = self._update_section_path(
                        section_path,
                        section_levels,
                        heading_level,
                        normalized_heading,
                    )
                    text = text[len(normalized_heading) :].strip()
                if not text:
                    continue

            heading_level = self._heading_level(item, text)
            if heading_level is not None:
                if units:
                    sections.append((section_path, units))
                    units = []
                section_path, section_levels = self._update_section_path(
                    section_path,
                    section_levels,
                    heading_level,
                    text,
                )
                continue

            for part in self._split_oversized_text(text):
                units.append(
                    _TextUnit(
                        text=part,
                        item_id=item.item_id,
                        page_idx=item.page_idx,
                        order_idx=item.order_idx,
                    )
                )

        if units:
            sections.append((section_path, units))
        return sections

    @staticmethod
    def _metadata_headings(item: ContentItem) -> tuple[tuple[int, str], ...]:
        headings: list[tuple[int, str]] = []
        values = item.metadata.get("headings") or []
        if not isinstance(values, list):
            return ()
        for value in values:
            if not isinstance(value, dict):
                continue
            text = str(value.get("text") or "").strip()
            try:
                level = int(value.get("level") or 0)
            except (TypeError, ValueError):
                continue
            if text and level > 0:
                headings.append((level, text))
        return tuple(headings)

    def _chunk_section(
        self,
        document: ParsedDocument,
        section_path: tuple[str, ...],
        units: list[_TextUnit],
        first_chunk_index: int,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        cursor = 0
        overlap: list[_TextUnit] = []

        while cursor < len(units):
            selected = list(overlap)
            added_units = 0

            while cursor < len(units):
                candidate = [*selected, units[cursor]]
                candidate_tokens = self._count_units(candidate)
                if candidate_tokens > self.config.max_tokens:
                    if added_units == 0 and selected:
                        selected = []
                        overlap = []
                        continue
                    break

                selected.append(units[cursor])
                cursor += 1
                added_units += 1
                if candidate_tokens >= self.config.target_tokens:
                    break

            if added_units == 0:
                raise RuntimeError("Text chunker could not make forward progress")

            text = self._join_units(selected)
            chunk_index = first_chunk_index + len(chunks)
            source_item_ids = tuple(
                dict.fromkeys(unit.item_id for unit in selected)
            )
            chunks.append(
                TextChunk(
                    chunk_id=stable_id(
                        "chunk",
                        document.document_id,
                        chunk_index,
                        section_path,
                        source_item_ids,
                        text,
                    ),
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=text,
                    token_count=len(self.tokenizer.encode(text)),
                    page_start=min(unit.page_idx for unit in selected),
                    page_end=max(unit.page_idx for unit in selected),
                    source_order_start=min(unit.order_idx for unit in selected),
                    source_item_ids=source_item_ids,
                    section_path=section_path,
                    metadata={
                        "tokenizer": self.tokenizer.name,
                        "target_tokens": self.config.target_tokens,
                        "max_tokens": self.config.max_tokens,
                        "overlap_tokens": self.config.overlap_tokens,
                    },
                )
            )
            overlap = (
                self._build_overlap(selected)
                if cursor < len(units)
                else []
            )

        return chunks

    def _split_oversized_text(self, text: str) -> list[str]:
        if len(self.tokenizer.encode(text)) <= self.config.max_tokens:
            return [text]

        pieces: list[str] = []
        pending: list[str] = []
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY_RE.split(text)
            if sentence.strip()
        ]
        for sentence in sentences:
            sentence_tokens = list(self.tokenizer.encode(sentence))
            if len(sentence_tokens) > self.config.max_tokens:
                if pending:
                    pieces.append(" ".join(pending))
                    pending = []
                for start in range(0, len(sentence_tokens), self.config.max_tokens):
                    part = self.tokenizer.decode(
                        sentence_tokens[start : start + self.config.max_tokens]
                    ).strip()
                    if part:
                        pieces.append(part)
                continue

            candidate = " ".join([*pending, sentence])
            if pending and len(self.tokenizer.encode(candidate)) > self.config.max_tokens:
                pieces.append(" ".join(pending))
                pending = [sentence]
            else:
                pending.append(sentence)

        if pending:
            pieces.append(" ".join(pending))
        return pieces

    def _build_overlap(self, units: list[_TextUnit]) -> list[_TextUnit]:
        remaining = self.config.overlap_tokens
        if remaining == 0:
            return []

        overlap: list[_TextUnit] = []
        for unit in reversed(units):
            tokens = list(self.tokenizer.encode(unit.text))
            if len(tokens) <= remaining:
                overlap.insert(0, unit)
                remaining -= len(tokens)
                if remaining == 0:
                    break
                continue

            tail = self.tokenizer.decode(tokens[-remaining:]).strip()
            if tail:
                overlap.insert(
                    0,
                    _TextUnit(
                        text=tail,
                        item_id=unit.item_id,
                        page_idx=unit.page_idx,
                        order_idx=unit.order_idx,
                    ),
                )
            break
        return overlap

    def _count_units(self, units: list[_TextUnit]) -> int:
        return len(self.tokenizer.encode(self._join_units(units)))

    @staticmethod
    def _join_units(units: list[_TextUnit]) -> str:
        return "\n\n".join(unit.text for unit in units).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text.strip())
        normalized = re.sub(r"\s*\n+\s*", " ", normalized)
        return re.sub(r"[ \t]+", " ", normalized).strip()

    @staticmethod
    def _heading_level(item: ContentItem, text: str) -> int | None:
        explicit_level = int(item.metadata.get("text_level") or 0)
        if explicit_level > 0:
            return explicit_level
        if len(text) > 160 or len(text.split()) > 16:
            return None

        numbered = _NUMBERED_HEADING_RE.match(text)
        if numbered:
            label = numbered.group("label")
            title = numbered.group("title")
            if label[0].isalpha():
                title_letters = "".join(
                    character for character in title if character.isalpha()
                )
                if (
                    not title_letters
                    or title_letters.upper() != title_letters
                    or "=" in title
                ):
                    return None
            return label.count(".") + 1

        if text.upper() in _STANDALONE_HEADINGS:
            return 1
        letters = "".join(character for character in text if character.isalpha())
        if (
            len(text.split()) >= 2
            and len(letters) >= 4
            and letters.upper() == letters
            and "=" not in text
        ):
            return 1
        return None

    @staticmethod
    def _update_section_path(
        current: tuple[str, ...],
        current_levels: tuple[int, ...],
        level: int,
        heading: str,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        keep = len(current_levels)
        while keep > 0 and current_levels[keep - 1] >= level:
            keep -= 1
        return (
            (*current[:keep], heading),
            (*current_levels[:keep], level),
        )

    @staticmethod
    def _find_repeated_boilerplate(document: ParsedDocument) -> set[str]:
        pages_by_text: dict[str, set[int]] = {}
        for item in document.items:
            if item.type != ContentType.TEXT:
                continue
            text = TextChunker._normalize_text(item.text)
            if not text or len(text) > 160:
                continue
            pages_by_text.setdefault(text, set()).add(item.page_idx)
        return {
            text
            for text, pages in pages_by_text.items()
            if len(pages) >= 3
        }
