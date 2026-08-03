from __future__ import annotations

import base64
from pathlib import Path

import fitz

from my_mrag.config import Settings
from my_mrag.pipeline import IngestionPipeline
from my_mrag.schemas import ContentType


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Reason Plan ReAct research paper")
    page.insert_image(fitz.Rect(72, 100, 180, 208), stream=_PNG_1X1)
    page.insert_text((72, 225), "Figure 1: Planner and executor architecture")
    document.save(path)
    document.close()


def _create_two_column_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((230, 60), "Full Width Title")
    page.insert_text((72, 120), "Left paragraph one")
    page.insert_text((72, 150), "Left paragraph two")
    page.insert_text((330, 120), "Right paragraph one")
    page.insert_text((330, 150), "Right paragraph two")
    document.save(path)
    document.close()


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


def test_pipeline_extracts_text_image_and_caption(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path)

    pipeline = IngestionPipeline(_settings(tmp_path))
    document, stored_path = pipeline.ingest(pdf_path)

    assert stored_path.is_file()
    assert document.page_count == 1
    assert any(
        item.type == ContentType.TEXT and "Reason Plan ReAct" in item.text
        for item in document.items
    )
    image = next(item for item in document.items if item.type == ContentType.IMAGE)
    assert image.asset_path is not None
    assert Path(image.asset_path).is_file()
    assert image.captions == ["Figure 1: Planner and executor architecture"]


def test_persisted_document_round_trips(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path)
    pipeline = IngestionPipeline(_settings(tmp_path))

    document, _ = pipeline.ingest(pdf_path)
    loaded = pipeline.load(document.document_id)

    assert loaded.to_dict() == document.to_dict()


def test_parser_preserves_two_column_reading_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "columns.pdf"
    _create_two_column_pdf(pdf_path)
    pipeline = IngestionPipeline(_settings(tmp_path))

    document, _ = pipeline.ingest(pdf_path)
    texts = [item.text for item in document.items if item.type == ContentType.TEXT]

    assert texts == [
        "Full Width Title",
        "Left paragraph one",
        "Left paragraph two",
        "Right paragraph one",
        "Right paragraph two",
    ]
