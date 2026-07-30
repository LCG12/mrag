from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from my_mrag.schemas import ContentItem, ContentType, ParsedDocument


class Tokenizer(Protocol):
    def encode(self, text: str) -> Sequence[object]:
        ...

    def decode(self, tokens: Sequence[object]) -> str:
        ...


@dataclass(frozen=True)
class ContextConfig:
    """Controls how neighboring content is selected for a multimodal item."""

    context_window: int = 1
    context_mode: str = "page"
    max_context_tokens: int = 2000
    fallback_characters_per_token: int = 4
    include_headers: bool = True
    include_captions: bool = True
    filter_content_types: tuple[ContentType, ...] = (ContentType.TEXT,)

    def __post_init__(self) -> None:
        if self.context_window < 0:
            raise ValueError("context_window cannot be negative")
        if self.context_mode not in {"page", "chunk"}:
            raise ValueError("context_mode must be 'page' or 'chunk'")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.fallback_characters_per_token <= 0:
            raise ValueError("fallback_characters_per_token must be positive")


class ContextExtractor:
    """Extract page- or chunk-level context from our normalized document."""

    def __init__(
        self,
        config: ContextConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer

    def extract_context(
        self,
        content_source: ParsedDocument | Sequence[ContentItem],
        current_item: ContentItem | str,
    ) -> str:
        items = (
            content_source.items
            if isinstance(content_source, ParsedDocument)
            else list(content_source)
        )
        if not items:
            return ""

        target = self._resolve_current_item(items, current_item)
        if self.config.context_mode == "chunk":
            context = self._extract_chunk_context(items, target)
        else:
            context = self._extract_page_context(items, target)
        return self._truncate_context(context)

    @staticmethod
    def _resolve_current_item(
        items: Sequence[ContentItem],
        current_item: ContentItem | str,
    ) -> ContentItem:
        item_id = (
            current_item.item_id
            if isinstance(current_item, ContentItem)
            else current_item
        )
        for item in items:
            if item.item_id == item_id:
                return item
        raise KeyError(f"Content item not found: {item_id}")

    def _extract_page_context(
        self,
        items: Sequence[ContentItem],
        current_item: ContentItem,
    ) -> str:
        start_page = max(0, current_item.page_idx - self.config.context_window)
        end_page = current_item.page_idx + self.config.context_window
        parts: list[str] = []

        for item in sorted(items, key=lambda value: value.order_idx):
            if (
                start_page <= item.page_idx <= end_page
                and item.type in self.config.filter_content_types
            ):
                text = self._extract_text(item)
                if not text:
                    continue
                if item.page_idx != current_item.page_idx:
                    text = f"[Page {item.page_idx + 1}] {text}"
                parts.append(text)
        return "\n".join(parts)

    def _extract_chunk_context(
        self,
        items: Sequence[ContentItem],
        current_item: ContentItem,
    ) -> str:
        ordered = sorted(items, key=lambda value: value.order_idx)
        current_index = next(
            index
            for index, item in enumerate(ordered)
            if item.item_id == current_item.item_id
        )
        start = max(0, current_index - self.config.context_window)
        end = min(len(ordered), current_index + self.config.context_window + 1)
        parts: list[str] = []

        for index in range(start, end):
            item = ordered[index]
            if (
                index != current_index
                and item.type in self.config.filter_content_types
            ):
                text = self._extract_text(item)
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _extract_text(self, item: ContentItem) -> str:
        if item.type == ContentType.TEXT:
            text = item.text.strip()
            text_level = int(item.metadata.get("text_level") or 0)
            if text and self.config.include_headers and text_level > 0:
                return f"{'#' * text_level} {text}"
            return text

        if (
            item.type in {ContentType.IMAGE, ContentType.TABLE}
            and self.config.include_captions
            and item.captions
        ):
            label = "Image" if item.type == ContentType.IMAGE else "Table"
            return f"[{label}: {', '.join(item.captions)}]"

        if item.type == ContentType.EQUATION:
            return item.text.strip()
        return ""

    def _truncate_context(self, context: str) -> str:
        if not context:
            return ""

        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= self.config.max_context_tokens:
                return context
            truncated = self.tokenizer.decode(
                tokens[: self.config.max_context_tokens]
            )
        else:
            character_limit = (
                self.config.max_context_tokens
                * self.config.fallback_characters_per_token
            )
            if len(context) <= character_limit:
                return context
            truncated = context[:character_limit]

        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        if last_period > len(truncated) * 0.8:
            return truncated[: last_period + 1]
        if last_newline > len(truncated) * 0.8:
            return truncated[:last_newline]
        return truncated.rstrip() + "..."
