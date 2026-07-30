from __future__ import annotations

from my_mrag import prompts
from my_mrag.processors.base import BaseModalProcessor, format_list
from my_mrag.schemas import (
    AnalysisRequest,
    ContentItem,
    ContentType,
    ParsedDocument,
)


class TableModalProcessor(BaseModalProcessor):
    content_type = ContentType.TABLE

    def prepare(
        self,
        document: ParsedDocument,
        item: ContentItem,
    ) -> AnalysisRequest:
        self._validate_item(document, item)
        body = item.text.strip()
        if not body:
            raise ValueError(f"Table item has no structured body: {item.item_id}")

        context = self.context_extractor.extract_context(document, item)
        prompt = prompts.TABLE_PROMPT.format(
            context=context or "None",
            image_path=item.asset_path or "None",
            captions=format_list(item.captions),
            body=body,
            footnotes=format_list(item.footnotes),
            response_schema=prompts.JSON_RESPONSE_SCHEMA.format(
                entity_type=item.type.value
            ),
        )
        return AnalysisRequest(
            item_id=item.item_id,
            document_id=document.document_id,
            content_type=item.type,
            system_prompt=prompts.TABLE_SYSTEM,
            prompt=prompt,
            context=context,
        )

    def build_chunk(
        self,
        item: ContentItem,
        context: str,
        description: str,
    ) -> str:
        return prompts.TABLE_CHUNK.format(
            image_path=item.asset_path or "None",
            captions=format_list(item.captions),
            body=item.text.strip(),
            footnotes=format_list(item.footnotes),
            description=description,
        ).strip()
