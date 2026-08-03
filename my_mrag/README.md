# my-mrag

`my-mrag` is a learning-oriented reimplementation of the core ideas in
RAG-Anything. We build the pipeline one layer at a time instead of treating the
upstream project as a black box.

The first milestone implements:

1. PDF parsing with page and bounding-box metadata.
2. A unified schema for text, image, table, and equation content.
3. Image and table asset extraction.
4. Two-column research-paper reading order.
5. JSON persistence for the normalized document.

The second milestone now implements the multimodal understanding layer:

1. `ContextExtractor` with page-window and chunk-window modes.
2. Context filtering, heading preservation, caption support, and truncation.
3. `ImageModalProcessor`, `TableModalProcessor`, and
   `EquationModalProcessor`.
4. A shared JSON result schema containing a detailed description and entity
   information.
5. Multimodal chunk construction and JSON persistence for the later LightRAG
   indexing stage.

The third milestone implements LightRAG indexing:

1. Convert every `chunk_text` into a LightRAG text chunk and chunk vector.
2. Convert every `entity_info` into a graph node and entity vector.
3. Create a stable source-document node.
4. Create a `belongs_to` edge from each multimodal entity to its source
   document, including a relationship vector.
5. Use LightRAG's public `ainsert_custom_kg()` API rather than its internal
   storage attributes.

The fourth milestone implements retrieval-ready text chunking:

1. Remove repeated running headers and standalone page numbers.
2. Preserve research-paper section boundaries and hierarchical section paths.
3. Merge ordered text blocks toward 600 tokens without exceeding 800 tokens.
4. Carry a 100-token overlap between adjacent chunks in the same section.
5. Preserve source item IDs, page ranges, reading order, and tokenizer metadata.
6. Persist chunks under `data/chunks/` and index them alongside multimodal
   analysis chunks.

The fifth milestone implements source-resolved hybrid retrieval:

1. Embed each query once and reuse it across chunk, entity, and relationship
   vector searches.
2. Resolve LightRAG internal chunk IDs back to local text chunks or multimodal
   items.
3. Expand vector-matched entities through graph neighbors and relationships.
4. Merge duplicate evidence while preserving every retrieval channel.
5. Return source document, page range, section path, asset path, and caption
   metadata for citations and later answer generation.

The sixth milestone implements our own text knowledge extraction:

1. Build an evidence-grounded entity and relationship prompt for every text
   chunk instead of calling LightRAG's built-in extraction pipeline.
2. Validate strict JSON, canonicalize relationship endpoints, and retry invalid
   model responses.
3. Persist readable per-chunk results under `data/knowledge/` for inspection.
4. Merge repeated entities and relationships across overlapping chunks.
5. Insert the resulting nodes, edges, entity vectors, and relationship vectors
   through LightRAG's public custom-KG API.

The lightweight PyMuPDF parser detects tables that expose sufficient ruling
lines. Borderless academic tables and equation recognition will be handled by
the next parser milestone, where we add a layout-model parser behind the same
`DocumentParser` interface.

BM25 is not part of this project. The normalized multimodal content is indexed
into LightRAG for vector and knowledge-graph retrieval. A later milestone will
add the RP-ReAct planner/executor layer.

## Run

```powershell
conda run -n mrag python -m pip install -e ".[graph,dev]"
conda run -n mrag python main.py parse ..\Reason-Plan-ReAct.pdf
conda run -n mrag python main.py inspect <document_id>
conda run -n mrag python main.py chunk <document_id>
conda run -n mrag python main.py extract-kg <document_id>
conda run -n mrag python main.py prepare <document_id> --full-prompts
conda run -n mrag python main.py retrieve "<question>" --document-id <document_id>
```

Parsed documents are stored under `data/parsed/`. Extracted images and table
crops are stored under `data/assets/<document_id>/`. Prepared model requests
and completed analyses are stored under `data/analysis/`. Retrieval-ready text
chunks are stored under `data/chunks/`.
Text entity and relationship extractions are stored under `data/knowledge/`.

`prepare` does not call a model. It is the recommended learning command because
it exposes the exact context, prompt, and image path produced for each item.

To execute table and equation analysis with the configured DeepSeek-compatible
text endpoint:

```powershell
conda run -n mrag python main.py analyze <document_id> --type table --type equation
```

Image analysis requires a genuinely vision-capable OpenAI-compatible endpoint.
Configure `VISION_API_KEY`, `VISION_BASE_URL`, and `VISION_MODEL`, then run:

```powershell
conda run -n mrag python main.py analyze <document_id> --type image
```

The configured DeepSeek text model is deliberately not reused as a vision
model. A model can only receive `image_paths` when its endpoint supports image
input.

The local `qwen3-embedding-0.6b` model is the default embedding backend:

```dotenv
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_PATH=./qwen3-embedding-0.6b
EMBEDDING_DEVICE=cuda
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=4
EMBEDDING_QUERY_PROMPT_NAME=query
EMBEDDING_MAX_TOKENS=8192
```

Qwen3 uses last-token pooling and normalized vectors. LightRAG sends
`context=document` while indexing and `context=query` while searching. The
adapter therefore leaves documents unchanged and automatically applies the
model's built-in `query` instruction to search queries.

Verify the local model, CUDA device, vector shape, and sample ranking:

```powershell
conda run -n mrag python main.py embedding-check
```

Then index the completed analysis:

```powershell
conda run -n mrag python main.py chunk <document_id>
conda run -n mrag python main.py extract-kg <document_id> --concurrency 2
conda run -n mrag python main.py index <document_id>
```

For a small real-model check before extracting every chunk:

```powershell
conda run -n mrag python main.py extract-kg <document_id> --limit 1
```

Completed chunks are skipped on later runs. Use `--force` to replace their
saved extraction. Use repeated `--chunk-id <id>` arguments to target specific
chunks.

LightRAG data is persisted under `data/lightrag/`. The indexing mapping is:

| my-mrag value | LightRAG destination |
| --- | --- |
| `TextChunk.index_text()` | `text_chunks` and `chunks_vdb` |
| `KnowledgeEntity` | graph node and `entities_vdb` |
| `KnowledgeRelationship` | graph edge and `relationships_vdb` |
| `ModalAnalysis.chunk_text` | `text_chunks` and `chunks_vdb` |
| `EntityInfo` | graph node and `entities_vdb` |
| `belongs_to` relation | graph edge and `relationships_vdb` |
| `ParsedDocument.source_path` | citation `file_path` |

Inspect retrieval before connecting an answer model:

```powershell
conda run -n mrag python main.py retrieve "How is the dual graph constructed?" --document-id <document_id> --top-k 5
```

The output separates chunks, entities, and relationships. Chunk hits include
their vector score, retrieval channels, one-based page range, section path,
and multimodal asset metadata when available.

On a fresh Windows environment with an RTX 50-series GPU, install a CUDA build
of PyTorch before installing the local embedding extra:

```powershell
conda run -n mrag python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
conda run -n mrag python -m pip install -e ".[graph,local-embeddings,dev]"
```

Set `EMBEDDING_PROVIDER=openai` to use the remote
`EMBEDDING_API_KEY/BASE_URL/MODEL` settings instead.
