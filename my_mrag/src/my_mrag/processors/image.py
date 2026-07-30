from __future__ import annotations

from pathlib import Path

from my_mrag import prompts
from my_mrag.processors.base import (
    BaseModalProcessor,
    format_list,
    format_section_path,
)
from my_mrag.schemas import (
    AnalysisRequest,
    ContentItem,
    ContentType,
    ParsedDocument,
)


class ImageModalProcessor(BaseModalProcessor):
    content_type = ContentType.IMAGE

    def prepare(
        self,
        document: ParsedDocument,
        item: ContentItem,
    ) -> AnalysisRequest:
        self._validate_item(document, item)
        if not item.asset_path:
            raise ValueError(f"Image item has no asset path: {item.item_id}")
        image_path = Path(item.asset_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image asset not found: {image_path}")

        context = self.context_extractor.extract_context(document, item)
        prompt = prompts.IMAGE_PROMPT.format(
            context=context or "None",
            section_path=format_section_path(item),
            image_path=item.asset_path,
            captions=format_list(item.captions),
            footnotes=format_list(item.footnotes),
            response_schema=prompts.JSON_RESPONSE_SCHEMA.format(
                entity_type=item.type.value
            ),
        )
        return AnalysisRequest(
            item_id=item.item_id,
            document_id=document.document_id,
            content_type=item.type,
            system_prompt=prompts.IMAGE_SYSTEM,
            prompt=prompt,
            context=context,
            image_paths=(item.asset_path,),
        )

    def build_chunk(
        self,
        item: ContentItem,
        context: str,
        description: str,
    ) -> str:
        return prompts.IMAGE_CHUNK.format(
            section_path=format_section_path(item),
            context=context or "None",
            image_path=item.asset_path or "None",
            captions=format_list(item.captions),
            footnotes=format_list(item.footnotes),
            description=description,
        ).strip()
