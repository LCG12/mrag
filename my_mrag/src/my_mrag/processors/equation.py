from __future__ import annotations

from my_mrag import prompts
from my_mrag.processors.base import BaseModalProcessor
from my_mrag.schemas import (
    AnalysisRequest,
    ContentItem,
    ContentType,
    ParsedDocument,
)


class EquationModalProcessor(BaseModalProcessor):
    content_type = ContentType.EQUATION

    def prepare(
        self,
        document: ParsedDocument,
        item: ContentItem,
    ) -> AnalysisRequest:
        self._validate_item(document, item)
        equation = item.text.strip()
        if not equation:
            raise ValueError(f"Equation item has no equation text: {item.item_id}")

        equation_format = str(
            item.metadata.get("equation_format")
            or item.metadata.get("format")
            or "latex"
        )
        context = self.context_extractor.extract_context(document, item)
        prompt = prompts.EQUATION_PROMPT.format(
            context=context or "None",
            equation=equation,
            equation_format=equation_format,
            response_schema=prompts.JSON_RESPONSE_SCHEMA.format(
                entity_type=item.type.value
            ),
        )
        return AnalysisRequest(
            item_id=item.item_id,
            document_id=document.document_id,
            content_type=item.type,
            system_prompt=prompts.EQUATION_SYSTEM,
            prompt=prompt,
            context=context,
        )

    def build_chunk(
        self,
        item: ContentItem,
        context: str,
        description: str,
    ) -> str:
        equation_format = str(
            item.metadata.get("equation_format")
            or item.metadata.get("format")
            or "latex"
        )
        return prompts.EQUATION_CHUNK.format(
            equation=item.text.strip(),
            equation_format=equation_format,
            description=description,
        ).strip()
