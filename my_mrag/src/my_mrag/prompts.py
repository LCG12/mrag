IMAGE_SYSTEM = (
    "You are an expert image analyst. Provide detailed, accurate descriptions."
)
TABLE_SYSTEM = (
    "You are an expert data analyst. Analyze tables using specific headers, "
    "values, comparisons, and trends."
)
EQUATION_SYSTEM = (
    "You are an expert mathematician. Explain equations precisely in the "
    "document's scientific context."
)

JSON_RESPONSE_SCHEMA = """Return only one JSON object:
{{
  "detailed_description": "detailed analysis useful for retrieval",
  "entity_info": {{
    "entity_name": "short semantic name, not a file name or figure number",
    "entity_type": "{entity_type}",
    "summary": "concise summary, at most 100 words"
  }}
}}"""

IMAGE_PROMPT = """Analyze the image in detail.
Describe its composition, visible text, objects, diagram or chart structure,
relationships between elements, and its meaning in the paper.

Surrounding context:
{context}

Document details:
- Section path: {section_path}
- Image path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

{response_schema}"""

TABLE_PROMPT = """Analyze the table.
Explain its structure, headers, important values, patterns, comparisons,
trends, and how the evidence supports the surrounding discussion.

Surrounding context:
{context}

Table details:
- Image path: {image_path}
- Captions: {captions}
- Body:
{body}
- Footnotes: {footnotes}

{response_schema}"""

EQUATION_PROMPT = """Analyze the mathematical equation.
Explain its mathematical meaning, variables, operations, assumptions,
application domain, significance, and role in the surrounding discussion.

Surrounding context:
{context}

Equation details:
- Equation: {equation}
- Format: {equation_format}

{response_schema}"""

IMAGE_CHUNK = """Image Content Analysis:
- Section Path: {section_path}
- Neighbor Text: {context}
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Visual Analysis: {description}"""

TABLE_CHUNK = """Table Analysis:
- Image Path: {image_path}
- Caption: {captions}
- Structure:
{body}
- Footnotes: {footnotes}

Analysis: {description}"""

EQUATION_CHUNK = """Mathematical Equation Analysis:
- Equation: {equation}
- Format: {equation_format}

Mathematical Analysis: {description}"""


KNOWLEDGE_EXTRACTION_SYSTEM = (
    "You are a scientific knowledge graph curator. Extract only entities and "
    "relationships explicitly supported by the supplied research-paper text. "
    "Return strict JSON and do not add commentary."
)

KNOWLEDGE_EXTRACTION_PROMPT = """Extract a compact knowledge graph from this text chunk.

Document context:
- Chunk ID: {chunk_id}
- Pages: {page_start}-{page_end}
- Section: {section_path}

<source_text>
{text}
</source_text>

Rules:
1. Extract at most {max_entities} important entities. Prefer named methods,
   models, datasets, tasks, metrics, components, concepts, and findings.
2. Preserve official names and use the same canonical name everywhere.
3. Descriptions must state what the source text says, not outside knowledge.
4. Extract at most {max_relationships} explicit, useful relationships.
5. Every relationship endpoint must exactly match an entity_name in entities.
6. Do not create relationships for mere co-occurrence.
7. Weight is an evidence-strength score from 1.0 to 10.0.
8. Return only one JSON object matching this structure:
{{
  "entities": [
    {{
      "entity_name": "canonical entity name",
      "entity_type": "method|model|dataset|task|metric|component|concept|finding|organization|person|other",
      "description": "concise evidence-grounded description"
    }}
  ],
  "relationships": [
    {{
      "source_entity": "exact entity_name",
      "target_entity": "exact entity_name",
      "description": "how the source text relates them",
      "keywords": ["short_relation_label"],
      "weight": 5.0
    }}
  ]
}}

Use empty arrays when the chunk contains no meaningful scientific entities.
"""
