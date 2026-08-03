from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from my_mrag.config import Settings
from my_mrag.context import ContextConfig, ContextExtractor
from my_mrag.multimodal import MultimodalPipeline
from my_mrag.processors import (
    EquationModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)
from my_mrag.schemas import (
    AnalysisRequest,
    ContentItem,
    ContentType,
    ParsedDocument,
)


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class FakeModel:
    def __init__(self, model_name: str, entity_type: str):
        self.model_name = model_name
        self.entity_type = entity_type
        self.requests: list[AnalysisRequest] = []

    async def complete(self, request: AnalysisRequest) -> str:
        self.requests.append(request)
        return f"""```json
{{
  "detailed_description": "Detailed {request.content_type.value} analysis.",
  "entity_info": {{
    "entity_name": "Planner architecture",
    "entity_type": "{self.entity_type}",
    "summary": "A concise scientific summary."
  }}
}}
```"""


def _item(
    item_id: str,
    content_type: ContentType,
    page_idx: int,
    order_idx: int,
    *,
    text: str = "",
    asset_path: str | None = None,
    captions: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> ContentItem:
    return ContentItem(
        item_id=item_id,
        document_id="doc-1",
        type=content_type,
        page_idx=page_idx,
        order_idx=order_idx,
        text=text,
        asset_path=asset_path,
        captions=captions or [],
        metadata=metadata or {},
    )


def _document(items: list[ContentItem]) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_path="paper.pdf",
        parser="test",
        page_count=3,
        items=items,
    )


def _settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        parsed_dir=data_dir / "parsed",
        assets_dir=data_dir / "assets",
        analysis_dir=data_dir / "analysis",
        chunks_dir=data_dir / "chunks",
        knowledge_dir=data_dir / "knowledge",
        lightrag_dir=data_dir / "lightrag",
    )


def test_context_extractor_supports_page_and_chunk_modes() -> None:
    intro = _item("text-0", ContentType.TEXT, 0, 0, text="Introduction")
    before = _item("text-1", ContentType.TEXT, 1, 1, text="Planner context")
    image = _item("image-1", ContentType.IMAGE, 1, 2)
    after = _item("text-2", ContentType.TEXT, 1, 3, text="Executor context")
    conclusion = _item("text-3", ContentType.TEXT, 2, 4, text="Conclusion")
    document = _document([intro, before, image, after, conclusion])

    page_context = ContextExtractor().extract_context(document, image)
    assert "[Page 1] Introduction" in page_context
    assert "Planner context" in page_context
    assert "Executor context" in page_context
    assert "[Page 3] Conclusion" in page_context

    chunk_extractor = ContextExtractor(
        ContextConfig(context_mode="chunk", context_window=1)
    )
    assert chunk_extractor.extract_context(document, image) == (
        "Planner context\nExecutor context"
    )


def test_context_extractor_preserves_headers_and_truncates() -> None:
    header = _item(
        "header",
        ContentType.TEXT,
        0,
        0,
        text="Methods",
        metadata={"text_level": 2},
    )
    image = _item("image", ContentType.IMAGE, 0, 1)
    extractor = ContextExtractor(
        ContextConfig(context_window=0, max_context_tokens=2)
    )

    assert extractor.extract_context(_document([header, image]), image) == (
        "## Metho..."
    )


def test_modality_processors_prepare_expected_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(_PNG_1X1)
    context = _item("text", ContentType.TEXT, 0, 0, text="Model context")
    image = _item(
        "image",
        ContentType.IMAGE,
        0,
        1,
        asset_path=str(image_path),
        captions=["Figure 1: Planner"],
        metadata={"section_path": ["Method", "Architecture"]},
    )
    table = _item(
        "table",
        ContentType.TABLE,
        0,
        2,
        text="| Method | Score |\n| --- | --- |\n| RPR | 92 |",
        captions=["Table 1: Results"],
    )
    equation = _item(
        "equation",
        ContentType.EQUATION,
        0,
        3,
        text=r"p(a \mid s)",
        metadata={"equation_format": "latex"},
    )
    document = _document([context, image, table, equation])

    image_request = ImageModalProcessor().prepare(document, image)
    table_request = TableModalProcessor().prepare(document, table)
    equation_request = EquationModalProcessor().prepare(document, equation)

    assert image_request.image_paths == (str(image_path),)
    assert "Model context" in image_request.prompt
    assert "Figure 1: Planner" in image_request.prompt
    assert "Method > Architecture" in image_request.prompt
    assert "| RPR | 92 |" in table_request.prompt
    assert r"p(a \mid s)" in equation_request.prompt


def test_pipeline_processes_and_persists_all_modalities(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(_PNG_1X1)
    items = [
        _item("text", ContentType.TEXT, 0, 0, text="Surrounding text"),
        _item(
            "image",
            ContentType.IMAGE,
            0,
            1,
            asset_path=str(image_path),
        ),
        _item(
            "table",
            ContentType.TABLE,
            0,
            2,
            text="| A |\n| --- |\n| 1 |",
        ),
        _item(
            "equation",
            ContentType.EQUATION,
            0,
            3,
            text="x + y = z",
        ),
    ]
    document = _document(items)
    text_model = FakeModel("fake-text", "ignored")
    vision_model = FakeModel("fake-vision", "ignored")
    pipeline = MultimodalPipeline(
        settings=_settings(tmp_path),
        text_model=text_model,
        vision_model=vision_model,
    )

    analyses, stored_path = asyncio.run(
        pipeline.analyze_and_save(document)
    )

    assert stored_path.is_file()
    assert [item.content_type for item in analyses] == [
        ContentType.IMAGE,
        ContentType.TABLE,
        ContentType.EQUATION,
    ]
    assert analyses[0].entity_info.entity_type == "image"
    assert analyses[1].model_name == "fake-text"
    assert analyses[2].chunk_text.startswith("Mathematical Equation Analysis:")
    assert len(vision_model.requests) == 1
    assert len(text_model.requests) == 2
    assert pipeline.store.load_analyses("doc-1") == analyses
