from __future__ import annotations

from collections.abc import Iterable

from my_mrag.processors.base import BaseModalProcessor
from my_mrag.schemas import ContentType


class ProcessorRegistry:
    def __init__(self, processors: Iterable[BaseModalProcessor] = ()):
        self._processors: dict[ContentType, BaseModalProcessor] = {}
        for processor in processors:
            self.register(processor)

    def register(self, processor: BaseModalProcessor) -> None:
        self._processors[processor.content_type] = processor

    def get(self, content_type: ContentType) -> BaseModalProcessor:
        try:
            return self._processors[content_type]
        except KeyError as exc:
            raise KeyError(
                f"No processor registered for {content_type.value}"
            ) from exc

    @property
    def supported_types(self) -> tuple[ContentType, ...]:
        return tuple(self._processors)
