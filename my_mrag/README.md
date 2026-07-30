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

The lightweight PyMuPDF parser detects tables that expose sufficient ruling
lines. Borderless academic tables and equation recognition will be handled by
the next parser milestone, where we add a layout-model parser behind the same
`DocumentParser` interface.

BM25 is not part of this project. Later milestones will connect the normalized
multimodal content to LightRAG for vector and knowledge-graph retrieval, then
add the RP-ReAct planner/executor layer.

## Run

```powershell
python -m pip install -e .
python main.py parse ..\Reason-Plan-ReAct.pdf
python main.py inspect <document_id>
python main.py prepare <document_id> --full-prompts
```

Parsed documents are stored under `data/parsed/`. Extracted images and table
crops are stored under `data/assets/<document_id>/`. Prepared model requests
and completed analyses are stored under `data/analysis/`.

`prepare` does not call a model. It is the recommended learning command because
it exposes the exact context, prompt, and image path produced for each item.

To execute table and equation analysis with the configured DeepSeek-compatible
text endpoint:

```powershell
python -m pip install -e .[models]
python main.py analyze <document_id> --type table --type equation
```

Image analysis requires a genuinely vision-capable OpenAI-compatible endpoint.
Configure `VISION_API_KEY`, `VISION_BASE_URL`, and `VISION_MODEL`, then run:

```powershell
python main.py analyze <document_id> --type image
```

The configured DeepSeek text model is deliberately not reused as a vision
model. A model can only receive `image_paths` when its endpoint supports image
input.
