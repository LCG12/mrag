from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from my_mrag.config import Settings
from my_mrag.models import OpenAICompatibleModel
from my_mrag.multimodal import MultimodalPipeline
from my_mrag.pipeline import IngestionPipeline
from my_mrag.schemas import ContentType


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse research documents into the my-mrag multimodal schema."
    )
    parser.add_argument(
        "--data-dir",
        help="Storage directory. Defaults to MY_MRAG_DATA_DIR or ./data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_command = subparsers.add_parser("parse", help="Parse and persist a PDF.")
    parse_command.add_argument("file", type=Path)

    inspect_command = subparsers.add_parser(
        "inspect", help="Inspect a previously parsed document."
    )
    inspect_command.add_argument("document_id")

    prepare_command = subparsers.add_parser(
        "prepare",
        help="Build and persist multimodal model requests without calling a model.",
    )
    _add_multimodal_arguments(prepare_command)
    prepare_command.add_argument(
        "--full-prompts",
        action="store_true",
        help="Print complete prompts instead of short previews.",
    )

    analyze_command = subparsers.add_parser(
        "analyze",
        help="Call configured models and persist multimodal analyses.",
    )
    _add_multimodal_arguments(analyze_command)
    return parser


def _add_multimodal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("document_id")
    parser.add_argument(
        "--item-id",
        action="append",
        dest="item_ids",
        help="Only process this item. May be repeated.",
    )
    parser.add_argument(
        "--type",
        action="append",
        choices=[
            ContentType.IMAGE.value,
            ContentType.TABLE.value,
            ContentType.EQUATION.value,
        ],
        dest="content_types",
        help="Only process this modality. May be repeated.",
    )


def _content_types(values: list[str] | None) -> list[ContentType] | None:
    if not values:
        return None
    return [ContentType(value) for value in values]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_parser().parse_args()
    settings = Settings.load(args.data_dir)
    pipeline = IngestionPipeline(settings)

    if args.command == "parse":
        document, stored_path = pipeline.ingest(args.file)
        summary = {
            "document_id": document.document_id,
            "source_path": document.source_path,
            "page_count": document.page_count,
            "item_count": len(document.items),
            "counts": document.count_by_type(),
            "stored_path": str(stored_path),
        }
    elif args.command == "inspect":
        document = pipeline.load(args.document_id)
        summary = {
            "document_id": document.document_id,
            "source_path": document.source_path,
            "page_count": document.page_count,
            "item_count": len(document.items),
            "counts": document.count_by_type(),
            "sample_items": [
                {
                    "type": item.type.value,
                    "page_idx": item.page_idx,
                    "text": item.text[:160],
                    "asset_path": item.asset_path,
                    "captions": item.captions,
                }
                for item in document.items[:8]
            ],
        }
    elif args.command == "prepare":
        document = pipeline.load(args.document_id)
        multimodal = MultimodalPipeline(settings=settings)
        requests, stored_path = multimodal.prepare_and_save(
            document,
            item_ids=args.item_ids,
            content_types=_content_types(args.content_types),
        )
        preview_length = None if args.full_prompts else 600
        summary = {
            "document_id": document.document_id,
            "request_count": len(requests),
            "stored_path": str(stored_path),
            "requests": [
                {
                    "item_id": request.item_id,
                    "content_type": request.content_type.value,
                    "context": request.context,
                    "image_paths": list(request.image_paths),
                    "prompt": (
                        request.prompt
                        if preview_length is None
                        else request.prompt[:preview_length]
                    ),
                }
                for request in requests
            ],
        }
    else:
        document = pipeline.load(args.document_id)
        multimodal = MultimodalPipeline(
            settings=settings,
            text_model=OpenAICompatibleModel.from_env(
                "DEEPSEEK",
                required=False,
            ),
            vision_model=OpenAICompatibleModel.from_env(
                "VISION",
                required=False,
            ),
        )
        analyses, stored_path = asyncio.run(
            multimodal.analyze_and_save(
                document,
                item_ids=args.item_ids,
                content_types=_content_types(args.content_types),
            )
        )
        summary = {
            "document_id": document.document_id,
            "analysis_count": len(analyses),
            "stored_path": str(stored_path),
            "analyses": [
                {
                    "item_id": analysis.item_id,
                    "content_type": analysis.content_type.value,
                    "entity_info": {
                        "entity_name": analysis.entity_info.entity_name,
                        "entity_type": analysis.entity_info.entity_type,
                        "summary": analysis.entity_info.summary,
                    },
                    "description": analysis.detailed_description,
                    "model_name": analysis.model_name,
                }
                for analysis in analyses
            ],
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
