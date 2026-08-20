from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from my_mrag.agent import (
    AgentConfig,
    ExecutorConfig,
    PlannerConfig,
    ProxyExecutor,
    ResearchPlanner,
    RPReActAgent,
)
from my_mrag.answering import AnswerConfig, AnswerPipeline
from my_mrag.chunking import (
    ApproximateTokenizer,
    TextChunkConfig,
    TextChunker,
    load_text_tokenizer,
)
from my_mrag.config import Settings
from my_mrag.indexing import LightRAGIndexer
from my_mrag.knowledge import (
    KnowledgeExtractionConfig,
    KnowledgeExtractionPipeline,
)
from my_mrag.models import OpenAICompatibleModel
from my_mrag.memory import ConversationMemory, MemoryConfig
from my_mrag.multimodal import MultimodalPipeline
from my_mrag.pipeline import IngestionPipeline
from my_mrag.retrieval import RetrievalConfig, RetrievalPipeline
from my_mrag.schemas import ContentType
from my_mrag.storage import (
    JsonAnalysisStore,
    JsonConversationStore,
    JsonKnowledgeStore,
    JsonTextChunkStore,
)


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

    chunk_command = subparsers.add_parser(
        "chunk",
        help="Build and persist retrieval-ready chunks from parsed text.",
    )
    chunk_command.add_argument("document_id")
    chunk_command.add_argument("--target-tokens", type=int, default=600)
    chunk_command.add_argument("--max-tokens", type=int, default=800)
    chunk_command.add_argument("--overlap-tokens", type=int, default=100)
    chunk_command.add_argument(
        "--approximate-tokenizer",
        action="store_true",
        help="Do not load the configured local Hugging Face tokenizer.",
    )

    extract_kg_command = subparsers.add_parser(
        "extract-kg",
        help="Extract and persist entities and relationships from text chunks.",
    )
    extract_kg_command.add_argument("document_id")
    extract_kg_command.add_argument(
        "--chunk-id",
        action="append",
        dest="chunk_ids",
        help="Only extract this chunk. May be repeated.",
    )
    extract_kg_command.add_argument(
        "--limit",
        type=int,
        help="Process at most this many selected chunks.",
    )
    extract_kg_command.add_argument("--concurrency", type=int, default=2)
    extract_kg_command.add_argument("--retries", type=int, default=1)
    extract_kg_command.add_argument("--max-entities", type=int, default=12)
    extract_kg_command.add_argument(
        "--max-relationships",
        type=int,
        default=16,
    )
    extract_kg_command.add_argument(
        "--model-max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for each extraction request.",
    )
    extract_kg_command.add_argument(
        "--force",
        action="store_true",
        help="Re-extract chunks that already have saved results.",
    )

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

    index_command = subparsers.add_parser(
        "index",
        help="Write chunks, extracted knowledge, and analyses into LightRAG.",
    )
    index_command.add_argument("document_id")

    retrieve_command = subparsers.add_parser(
        "retrieve",
        help="Retrieve source-resolved chunk and graph evidence.",
    )
    retrieve_command.add_argument("query")
    retrieve_command.add_argument("--document-id")
    retrieve_command.add_argument("--top-k", type=int, default=8)
    retrieve_command.add_argument("--entity-top-k", type=int, default=5)
    retrieve_command.add_argument("--relationship-top-k", type=int, default=5)
    retrieve_command.add_argument("--graph-depth", type=int, default=1)
    retrieve_command.add_argument(
        "--full-content",
        action="store_true",
        help="Print complete retrieved chunk content.",
    )

    answer_command = subparsers.add_parser(
        "answer",
        help="Retrieve evidence and generate a source-cited answer.",
    )
    answer_command.add_argument("query")
    answer_command.add_argument("--document-id")
    answer_command.add_argument("--top-k", type=int, default=6)
    answer_command.add_argument("--entity-top-k", type=int, default=5)
    answer_command.add_argument("--relationship-top-k", type=int, default=5)
    answer_command.add_argument("--graph-depth", type=int, default=1)
    answer_command.add_argument(
        "--max-context-characters",
        type=int,
        default=24000,
        help="Maximum evidence characters sent to the answer model.",
    )

    agent_command = subparsers.add_parser(
        "agent",
        help="Plan retrieval steps, execute them, and answer with citations.",
    )
    agent_command.add_argument("query")
    agent_command.add_argument("--document-id")
    agent_command.add_argument(
        "--session-id",
        help="Persist and reuse conversation history under this session ID.",
    )
    agent_command.add_argument(
        "--memory-turns",
        type=int,
        default=4,
        help="Recent completed turns included as conversation context.",
    )
    agent_command.add_argument("--max-steps", type=int, default=3)
    agent_command.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="Maximum evidence-review and follow-up rounds; 0 disables review.",
    )
    agent_command.add_argument(
        "--max-followup-steps",
        type=int,
        default=1,
        help="Maximum new retrieval steps proposed by each review.",
    )
    agent_command.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Chunk results retained by each planned retrieval step.",
    )
    agent_command.add_argument("--entity-top-k", type=int, default=4)
    agent_command.add_argument("--relationship-top-k", type=int, default=4)
    agent_command.add_argument("--graph-depth", type=int, default=1)
    agent_command.add_argument("--concurrency", type=int, default=2)
    agent_command.add_argument(
        "--max-evidence-chunks",
        type=int,
        default=10,
        help="Maximum RRF-fused chunks sent to answer generation.",
    )
    agent_command.add_argument("--rrf-k", type=int, default=60)
    agent_command.add_argument(
        "--max-context-characters",
        type=int,
        default=24000,
    )

    memory_inspect_command = subparsers.add_parser(
        "memory-inspect",
        help="Inspect a persisted conversation session.",
    )
    memory_inspect_command.add_argument("session_id")

    embedding_command = subparsers.add_parser(
        "embedding-check",
        help="Load the configured embedding model and compare sample texts.",
    )
    embedding_command.add_argument(
        "--query",
        default="How does RP-ReAct separate planning from tool execution?",
    )
    embedding_command.add_argument(
        "--positive",
        default=(
            "RP-ReAct separates a high-level Reasoner Planner from "
            "low-level proxy agents that execute tool calls."
        ),
    )
    embedding_command.add_argument(
        "--negative",
        default=(
            "The paper reports evaluation results across several datasets."
        ),
    )
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
        sys.stdout.reconfigure(errors="backslashreplace")
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
    elif args.command == "chunk":
        document = pipeline.load(args.document_id)
        tokenizer = (
            ApproximateTokenizer()
            if args.approximate_tokenizer
            else load_text_tokenizer(settings)
        )
        chunker = TextChunker(
            config=TextChunkConfig(
                target_tokens=args.target_tokens,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens,
            ),
            tokenizer=tokenizer,
        )
        chunks = chunker.chunk(document)
        stored_path = JsonTextChunkStore(settings.chunks_dir).save(
            document.document_id,
            chunks,
        )
        token_counts = [chunk.token_count for chunk in chunks]
        summary = {
            "document_id": document.document_id,
            "chunk_count": len(chunks),
            "tokenizer": tokenizer.name,
            "token_counts": {
                "minimum": min(token_counts, default=0),
                "maximum": max(token_counts, default=0),
                "average": (
                    round(sum(token_counts) / len(token_counts), 2)
                    if token_counts
                    else 0
                ),
            },
            "stored_path": str(stored_path),
            "sample_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "section_path": list(chunk.section_path),
                    "pages": [chunk.page_start + 1, chunk.page_end + 1],
                    "token_count": chunk.token_count,
                    "source_item_ids": list(chunk.source_item_ids),
                    "text": chunk.text[:300],
                }
                for chunk in chunks[:3]
            ],
        }
    elif args.command == "extract-kg":
        document = pipeline.load(args.document_id)
        chunk_store = JsonTextChunkStore(settings.chunks_dir)
        if not chunk_store.exists(document.document_id):
            raise FileNotFoundError(
                f"No text chunks found for {document.document_id}. "
                "Run 'chunk' first."
            )
        chunks = chunk_store.load(document.document_id)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        selected_ids = set(args.chunk_ids or ())
        missing_ids = selected_ids - chunks_by_id.keys()
        if missing_ids:
            raise KeyError(f"Requested chunks not found: {sorted(missing_ids)}")
        selected = [
            chunk
            for chunk in chunks
            if not selected_ids or chunk.chunk_id in selected_ids
        ]

        knowledge_store = JsonKnowledgeStore(settings.knowledge_dir)
        existing = (
            knowledge_store.load(document.document_id)
            if knowledge_store.exists(document.document_id)
            else []
        )
        existing = [
            extraction
            for extraction in existing
            if extraction.chunk_id in chunks_by_id
        ]
        existing_ids = {extraction.chunk_id for extraction in existing}
        if not args.force:
            selected = [
                chunk for chunk in selected if chunk.chunk_id not in existing_ids
            ]
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("limit must be positive")
            selected = selected[: args.limit]

        merged = {item.chunk_id: item for item in existing}
        extracted = []
        if selected:
            if args.model_max_tokens <= 0:
                raise ValueError("model-max-tokens must be positive")
            model = OpenAICompatibleModel.from_env(
                "DEEPSEEK",
                max_tokens=args.model_max_tokens,
                json_mode=True,
                thinking=False,
            )
            assert model is not None
            extractor = KnowledgeExtractionPipeline(
                model,
                KnowledgeExtractionConfig(
                    max_entities=args.max_entities,
                    max_relationships=args.max_relationships,
                    concurrency=args.concurrency,
                    retries=args.retries,
                ),
            )

            async def run_extraction_batches() -> None:
                for start in range(0, len(selected), args.concurrency):
                    batch = selected[start : start + args.concurrency]
                    batch_results = await extractor.extract(batch)
                    extracted.extend(batch_results)
                    merged.update(
                        {item.chunk_id: item for item in batch_results}
                    )
                    checkpoint = sorted(
                        merged.values(),
                        key=lambda extraction: extraction.chunk_index,
                    )
                    knowledge_store.save(document.document_id, checkpoint)

            asyncio.run(run_extraction_batches())

        all_extractions = sorted(
            merged.values(),
            key=lambda extraction: extraction.chunk_index,
        )
        stored_path = knowledge_store.save(
            document.document_id,
            all_extractions,
        )
        summary = {
            "document_id": document.document_id,
            "selected_chunk_count": len(selected),
            "new_extraction_count": len(extracted),
            "saved_extraction_count": len(all_extractions),
            "entity_count": sum(
                len(extraction.entities)
                for extraction in all_extractions
            ),
            "relationship_count": sum(
                len(extraction.relationships)
                for extraction in all_extractions
            ),
            "stored_path": str(stored_path),
            "sample_extractions": [
                {
                    "chunk_id": extraction.chunk_id,
                    "chunk_index": extraction.chunk_index,
                    "model_name": extraction.model_name,
                    "entities": [
                        entity.entity_name for entity in extraction.entities
                    ],
                    "relationships": [
                        (
                            f"{relationship.source_entity} -> "
                            f"{relationship.target_entity}"
                        )
                        for relationship in extraction.relationships
                    ],
                }
                for extraction in extracted[:5]
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
    elif args.command == "analyze":
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
    elif args.command == "index":
        from my_mrag.lightrag_runtime import open_lightrag

        document = pipeline.load(args.document_id)
        analysis_store = JsonAnalysisStore(settings.analysis_dir)
        chunk_store = JsonTextChunkStore(settings.chunks_dir)
        knowledge_store = JsonKnowledgeStore(settings.knowledge_dir)
        analyses = (
            analysis_store.load_analyses(document.document_id)
            if analysis_store.exists(document.document_id)
            else []
        )
        text_chunks = (
            chunk_store.load(document.document_id)
            if chunk_store.exists(document.document_id)
            else []
        )
        knowledge = (
            knowledge_store.load(document.document_id)
            if knowledge_store.exists(document.document_id)
            else []
        )
        if not analyses and not text_chunks:
            raise FileNotFoundError(
                "No text chunks or multimodal analyses found for "
                f"{document.document_id}. Run 'chunk' and/or 'analyze' first."
            )

        async def run_index() -> dict[str, object]:
            async with open_lightrag(settings) as rag:
                report = await LightRAGIndexer(rag).index(
                    document,
                    analyses,
                    text_chunks,
                    knowledge,
                )
            return report.to_dict()

        summary = asyncio.run(run_index())
    elif args.command == "retrieve":
        from my_mrag.lightrag_runtime import open_lightrag

        if args.document_id:
            pipeline.load(args.document_id)

        async def run_retrieval() -> dict[str, object]:
            async with open_lightrag(settings) as rag:
                result = await RetrievalPipeline(
                    rag,
                    settings,
                    RetrievalConfig(
                        chunk_top_k=args.top_k,
                        entity_top_k=args.entity_top_k,
                        relationship_top_k=args.relationship_top_k,
                        final_top_k=args.top_k,
                        graph_depth=args.graph_depth,
                    ),
                ).retrieve(
                    args.query,
                    document_id=args.document_id,
                )
            payload = result.to_dict()
            if not args.full_content:
                for chunk in payload["chunks"]:
                    content = str(chunk["content"])
                    chunk["content"] = content[:600]
            payload["counts"] = {
                "chunks": len(result.chunks),
                "entities": len(result.entities),
                "relationships": len(result.relationships),
            }
            return payload

        summary = asyncio.run(run_retrieval())
    elif args.command == "answer":
        from my_mrag.lightrag_runtime import open_lightrag

        if args.document_id:
            pipeline.load(args.document_id)

        model = OpenAICompatibleModel.from_env(
            "DEEPSEEK",
            thinking=False,
        )
        assert model is not None

        async def run_answer() -> dict[str, object]:
            async with open_lightrag(settings) as rag:
                retriever = RetrievalPipeline(
                    rag,
                    settings,
                    RetrievalConfig(
                        chunk_top_k=args.top_k,
                        entity_top_k=args.entity_top_k,
                        relationship_top_k=args.relationship_top_k,
                        final_top_k=args.top_k,
                        graph_depth=args.graph_depth,
                    ),
                )
                result = await AnswerPipeline(
                    retriever,
                    model,
                    AnswerConfig(
                        max_context_characters=(
                            args.max_context_characters
                        )
                    ),
                ).answer(
                    args.query,
                    document_id=args.document_id,
                )
            return result.to_dict()

        summary = asyncio.run(run_answer())
    elif args.command == "agent":
        from my_mrag.lightrag_runtime import open_lightrag

        if args.document_id:
            pipeline.load(args.document_id)

        planner_model = OpenAICompatibleModel.from_env(
            "DEEPSEEK",
            json_mode=True,
            thinking=False,
        )
        answer_model = OpenAICompatibleModel.from_env(
            "DEEPSEEK",
            thinking=False,
        )
        assert planner_model is not None
        assert answer_model is not None

        async def run_agent() -> dict[str, object]:
            async with open_lightrag(settings) as rag:
                retriever = RetrievalPipeline(
                    rag,
                    settings,
                    RetrievalConfig(
                        chunk_top_k=args.top_k,
                        entity_top_k=args.entity_top_k,
                        relationship_top_k=args.relationship_top_k,
                        final_top_k=args.top_k,
                        graph_depth=args.graph_depth,
                    ),
                )
                agent = RPReActAgent(
                    ResearchPlanner(
                        planner_model,
                        PlannerConfig(
                            max_steps=args.max_steps,
                            max_followup_steps=args.max_followup_steps,
                        ),
                    ),
                    ProxyExecutor(
                        retriever,
                        ExecutorConfig(
                            concurrency=args.concurrency,
                            max_chunks=args.max_evidence_chunks,
                            rrf_k=args.rrf_k,
                        ),
                    ),
                    AnswerPipeline(
                        retriever,
                        answer_model,
                        AnswerConfig(
                            max_context_characters=(
                                args.max_context_characters
                            )
                        ),
                    ),
                    AgentConfig(max_replans=args.max_replans),
                    memory=(
                        ConversationMemory(
                            JsonConversationStore(
                                settings.data_dir / "memory"
                            ),
                            MemoryConfig(
                                context_turns=args.memory_turns
                            ),
                        )
                        if args.session_id
                        else None
                    ),
                )
                result = await agent.run(
                    args.query,
                    document_id=args.document_id,
                    session_id=args.session_id,
                )
            return result.to_dict()

        summary = asyncio.run(run_agent())
    elif args.command == "memory-inspect":
        memory_store = JsonConversationStore(settings.data_dir / "memory")
        session = memory_store.load(args.session_id)
        summary = session.to_dict()
        summary["stored_path"] = str(
            memory_store.path_for(args.session_id)
        )
    else:
        from my_mrag.embeddings import (
            build_embedding_model,
            load_embedding_settings,
        )

        async def run_embedding_check() -> dict[str, object]:
            import numpy as np

            embedding_settings = load_embedding_settings(settings)
            embedding_model = build_embedding_model(embedding_settings)
            query_vector = await embedding_model(
                [args.query],
                context="query",
            )
            document_vectors = await embedding_model(
                [args.positive, args.negative],
                context="document",
            )
            similarities = document_vectors @ query_vector[0]
            return {
                "config": embedding_settings.public_dict(),
                "resolved_device": getattr(
                    embedding_model,
                    "resolved_device",
                    "remote",
                ),
                "shape": list(query_vector.shape),
                "query_norm": round(
                    float(np.linalg.norm(query_vector[0])),
                    6,
                ),
                "positive_similarity": round(
                    float(similarities[0]),
                    6,
                ),
                "negative_similarity": round(
                    float(similarities[1]),
                    6,
                ),
                "ranking_is_correct": bool(
                    similarities[0] > similarities[1]
                ),
            }

        summary = asyncio.run(run_embedding_check())

    print(json.dumps(summary, ensure_ascii=False, indent=2))
