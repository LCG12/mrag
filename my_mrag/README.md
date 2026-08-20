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

The seventh milestone implements grounded answer generation:

1. Convert ranked chunks into stable `[S1]`, `[S2]`, ... evidence blocks.
2. Include the source file, one-based PDF pages, section path, modality,
   caption, and content in every citation target.
3. Use retrieved entities and relationships as navigation hints without
   treating graph summaries as independently citable evidence.
4. Generate an answer in the question's language and validate every citation.
5. Retry answers that omit citations, cite unknown sources, or cite graph hints
   instead of original chunks.

The eighth milestone implements the first bounded RP-ReAct agent loop:

1. A Reasoner Planner produces one to three evidence objectives as strict JSON.
2. A retrieval-only Proxy Executor translates each objective into a hybrid
   retrieval tool call and records the observation.
3. Independent retrieval steps run concurrently with a configurable limit.
4. Reciprocal Rank Fusion (RRF) merges and deduplicates chunk, entity, and
   relationship results across queries.
5. After observing fused source summaries, the planner decides whether to
   answer or add a focused retrieval step, within a strict replan limit.
6. The grounded answer stage consumes the fused evidence directly, without an
   extra hidden retrieval call.

The ninth milestone implements bounded persistent conversation memory:

1. Store completed questions, answers, document scope, timestamps, and only
   the sources actually cited by each answer.
2. Persist sessions atomically as readable JSON under `data/memory/`.
3. Inject a configurable number of recent turns into planning, evidence review,
   and answer generation for reference resolution.
4. Treat remembered text as quoted conversation context rather than citable
   research evidence or executable instructions.
5. Write a turn only after retrieval, generation, and citation validation all
   succeed.

The PyMuPDF parser detects ruled and Booktabs-style academic tables, as well as
displayed equations represented in the PDF text layer. Scanned pages and more
complex visual layouts remain candidates for a future layout/OCR parser behind
the same `DocumentParser` interface.

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
conda run -n mrag python main.py answer "<question>" --document-id <document_id>
conda run -n mrag python main.py agent "<complex-question>" --document-id <document_id>
conda run -n mrag python main.py agent "<follow-up>" --document-id <document_id> --session-id research-1
conda run -n mrag python main.py memory-inspect research-1
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

Inspect retrieval evidence directly:

```powershell
conda run -n mrag python main.py retrieve "How is the dual graph constructed?" --document-id <document_id> --top-k 5
```

The output separates chunks, entities, and relationships. Chunk hits include
their vector score, retrieval channels, one-based page range, section path,
and multimodal asset metadata when available.

Generate a grounded answer with the configured DeepSeek-compatible model:

```powershell
conda run -n mrag python main.py answer "How does RP-ReAct separate planning from execution?" --document-id <document_id> --top-k 6
```

The JSON output contains the Markdown answer, the cited `[S#]` IDs, and a
source map back to each chunk's document, page range, section, and modality.

For a multi-part question, run the planner/executor path:

```powershell
conda run -n mrag python main.py agent "How does RP-ReAct separate planning from execution, and when does that separation help or hurt?" --document-id <document_id> --max-steps 3 --max-replans 1
```

The output exposes the generated plan, every `retrieve_evidence` call, each
evidence-sufficiency review, fused evidence counts, and the final source-cited
answer. `stop_reason` distinguishes sufficient evidence from a reached review
limit. Set `--max-replans 0` for the faster plan-once path. The current proxy
executor has one reliable tool; later milestones can add paper comparison,
citation lookup, and calculation tools behind the same execution boundary.

Reuse the same session ID for follow-up questions:

```powershell
conda run -n mrag python main.py agent "How is CPS calculated?" --document-id <document_id> --session-id rp-react-study
conda run -n mrag python main.py agent "Why does it combine those two terms?" --document-id <document_id> --session-id rp-react-study
conda run -n mrag python main.py memory-inspect rp-react-study
```

Session IDs may contain letters, digits, underscores, and hyphens. By default,
the most recent four completed turns are included in model context; change this
with `--memory-turns`.

On a fresh Windows environment with an RTX 50-series GPU, install a CUDA build
of PyTorch before installing the local embedding extra:

```powershell
conda run -n mrag python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
conda run -n mrag python -m pip install -e ".[graph,local-embeddings,dev]"
```

Set `EMBEDDING_PROVIDER=openai` to use the remote
`EMBEDDING_API_KEY/BASE_URL/MODEL` settings instead.
