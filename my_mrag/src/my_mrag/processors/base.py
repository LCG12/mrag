from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from my_mrag.context import ContextExtractor
from my_mrag.models import AnalysisModel
from my_mrag.schemas import (
    AnalysisRequest,
    ContentItem,
    ContentType,
    EntityInfo,
    ModalAnalysis,
    ParsedDocument,
)
from my_mrag.utils import stable_id


class BaseModalProcessor(ABC):
    """Prepare model input and normalize model output for one modality."""

    content_type: ContentType

    def __init__(self, context_extractor: ContextExtractor | None = None):
        self.context_extractor = context_extractor or ContextExtractor()

    @abstractmethod
    def prepare(
        self,
        document: ParsedDocument,
        item: ContentItem,
    ) -> AnalysisRequest:
        ...

    @abstractmethod
    def build_chunk(
        self,
        item: ContentItem,
        context: str,
        description: str,
    ) -> str:
        ...

    async def process(
        self,
        document: ParsedDocument,
        item: ContentItem,
        model: AnalysisModel,
        *,
        entity_name: str | None = None,
    ) -> ModalAnalysis:
        request = self.prepare(document, item)
        response = await model.complete(request)
        description, entity_info = self._parse_response(
            response,
            item,
            entity_name=entity_name,
        )
        return ModalAnalysis(
            item_id=item.item_id,
            document_id=document.document_id,
            content_type=item.type,
            detailed_description=description,
            entity_info=entity_info,
            context=request.context,
            chunk_text=self.build_chunk(item, request.context, description),
            model_name=model.model_name,
        )

    def _validate_item(
        self,
        document: ParsedDocument,
        item: ContentItem,
    ) -> None:
        if item.document_id != document.document_id:
            raise ValueError(
                f"Item {item.item_id} does not belong to {document.document_id}"
            )
        if item.type != self.content_type:
            raise ValueError(
                f"{type(self).__name__} cannot process {item.type.value}"
            )

    def _parse_response(
        self,
        response: str,
        item: ContentItem,
        *,
        entity_name: str | None = None,
    ) -> tuple[str, EntityInfo]:
        cleaned = self._strip_thinking_tags(response)
        payload = self._parse_json_object(cleaned)
        if payload:
            description = str(payload.get("detailed_description") or "").strip()
            raw_entity = payload.get("entity_info")
            if description and isinstance(raw_entity, dict):
                name = entity_name or str(
                    raw_entity.get("entity_name") or ""
                ).strip()
                summary = str(raw_entity.get("summary") or "").strip()
                if name and summary:
                    return description, EntityInfo(
                        entity_name=name,
                        entity_type=self.content_type.value,
                        summary=summary,
                    )

        description = cleaned or self._fallback_description(item)
        fallback_name = entity_name or stable_id(
            self.content_type.value,
            item.item_id,
        )
        summary = " ".join(description.split())
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."
        return description, EntityInfo(
            entity_name=fallback_name,
            entity_type=self.content_type.value,
            summary=summary,
        )

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, object] | None:
        decoder = json.JSONDecoder()
        candidates = [
            match.group(1)
            for match in re.finditer(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
        ]
        candidates.append(text)

        for candidate in candidates:
            for start, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
        return None

    @staticmethod
    def _fallback_description(item: ContentItem) -> str:
        details = item.text.strip()
        if not details and item.captions:
            details = " ".join(item.captions)
        return details or f"Unparsed {item.type.value} content"


def format_list(values: list[str]) -> str:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(normalized) if normalized else "None"


def format_section_path(item: ContentItem) -> str:
    value = item.metadata.get("section_path")
    if isinstance(value, (list, tuple)):
        return " > ".join(str(part) for part in value if str(part).strip()) or "None"
    return str(value).strip() if value else "None"
