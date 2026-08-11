from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from my_mrag.config import Settings
from my_mrag.context import ContextExtractor
from my_mrag.models import AnalysisModel
from my_mrag.processors import (
    EquationModalProcessor,
    ImageModalProcessor,
    ProcessorRegistry,
    TableModalProcessor,
)
from my_mrag.schemas import (
    AnalysisRequest,
    ContentItem,
    ContentType,
    ModalAnalysis,
    ParsedDocument,
)
from my_mrag.storage import JsonAnalysisStore


class MultimodalPipeline:
    """Run context-aware processors over normalized non-text content."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        text_model: AnalysisModel | None = None,
        vision_model: AnalysisModel | None = None,
        context_extractor: ContextExtractor | None = None,
        registry: ProcessorRegistry | None = None,
    ):
        self.settings = settings or Settings.load()
        self.settings.ensure_directories()
        self.text_model = text_model
        self.vision_model = vision_model
        self.store = JsonAnalysisStore(self.settings.analysis_dir)

        extractor = context_extractor or ContextExtractor()
        self.registry = registry or ProcessorRegistry(
            [
                ImageModalProcessor(extractor),
                TableModalProcessor(extractor),
                EquationModalProcessor(extractor),
            ]
        )

    def prepare(
        self,
        document: ParsedDocument,
        *,
        item_ids: Iterable[str] | None = None,
        content_types: Iterable[ContentType] | None = None,
    ) -> list[AnalysisRequest]:
        return [
            self.registry.get(item.type).prepare(document, item)
            for item in self._select_items(
                document,
                item_ids=item_ids,
                content_types=content_types,
            )
        ]

    def prepare_and_save(
        self,
        document: ParsedDocument,
        *,
        item_ids: Iterable[str] | None = None,
        content_types: Iterable[ContentType] | None = None,
    ) -> tuple[list[AnalysisRequest], Path]:
        requests = self.prepare(
            document,
            item_ids=item_ids,
            content_types=content_types,
        )
        return requests, self.store.save_requests(document.document_id, requests)

    async def analyze(
        self,
        document: ParsedDocument,
        *,
        item_ids: Iterable[str] | None = None,
        content_types: Iterable[ContentType] | None = None,
    ) -> list[ModalAnalysis]:
        results: list[ModalAnalysis] = []
        for item in self._select_items(
            document,
            item_ids=item_ids,
            content_types=content_types,
        ):
            model = self._model_for(item)
            processor = self.registry.get(item.type)
            results.append(await processor.process(document, item, model))
        return results

    async def analyze_and_save(
        self,
        document: ParsedDocument,
        *,
        item_ids: Iterable[str] | None = None,
        content_types: Iterable[ContentType] | None = None,
    ) -> tuple[list[ModalAnalysis], Path]:
        analyses = await self.analyze(
            document,
            item_ids=item_ids,
            content_types=content_types,
        )
        items_by_id = {item.item_id: item for item in document.items}
        merged = {
            analysis.item_id: analysis
            for analysis in (
                self.store.load_analyses(document.document_id)
                if self.store.exists(document.document_id)
                else []
            )
            if analysis.item_id in items_by_id
        }
        merged.update({analysis.item_id: analysis for analysis in analyses})
        stored_analyses = sorted(
            merged.values(),
            key=lambda analysis: items_by_id[analysis.item_id].order_idx,
        )
        return analyses, self.store.save_analyses(
            document.document_id,
            stored_analyses,
        )

    def _select_items(
        self,
        document: ParsedDocument,
        *,
        item_ids: Iterable[str] | None,
        content_types: Iterable[ContentType] | None,
    ) -> list[ContentItem]:
        selected_ids = set(item_ids or ())
        selected_types = set(content_types or self.registry.supported_types)
        supported_types = set(self.registry.supported_types)
        unsupported = selected_types - supported_types
        if unsupported:
            names = ", ".join(sorted(value.value for value in unsupported))
            raise ValueError(f"Unsupported processor types: {names}")

        items = [
            item
            for item in sorted(
                document.items,
                key=lambda value: value.order_idx,
            )
            if item.type in selected_types
            and (not selected_ids or item.item_id in selected_ids)
        ]
        if selected_ids:
            missing = selected_ids - {item.item_id for item in items}
            if missing:
                raise KeyError(
                    f"Requested items not found or not multimodal: {sorted(missing)}"
                )
        return items

    def _model_for(self, item: ContentItem) -> AnalysisModel:
        if item.type == ContentType.IMAGE:
            if self.vision_model is None:
                raise RuntimeError(
                    "Image analysis requires a vision model configured with "
                    "VISION_API_KEY, VISION_BASE_URL, and VISION_MODEL"
                )
            return self.vision_model
        if self.text_model is None:
            raise RuntimeError(
                f"{item.type.value} analysis requires a text model configured "
                "with DEEPSEEK_API_KEY and DEEPSEEK_MODEL"
            )
        return self.text_model
