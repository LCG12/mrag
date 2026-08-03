import base64
from pathlib import Path

import fitz

from my_mrag.parsers import PyMuPDFParser
from my_mrag.schemas import ContentType


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

def test_real_pdf_text_and_tables(tmp_path: Path) -> None:
    pdf_path = Path("D:/projiect/sh/RAG-anything.pdf")
    parser = PyMuPDFParser(tmp_path / "assets")

    document = parser.parse(pdf_path)

    texts = [
        item for item in document.items
        if item.type == ContentType.TEXT
    ]
    tables = [
        item for item in document.items
        if item.type == ContentType.TABLE
    ]
    print(tables[1])
    assert len(texts) > 300
    assert len(tables) == 3

    full_text = "\n".join(item.text for item in texts)
    assert "RAG-Anything" in full_text

    for table in tables:
        assert table.text.strip()
        assert "|" in table.text
        assert table.asset_path
        assert Path(table.asset_path).is_file()