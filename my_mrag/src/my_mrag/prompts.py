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
