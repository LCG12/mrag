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
    equations = [
        item for item in document.items
        if item.type == ContentType.EQUATION
    ]
    assert len(texts) > 300
    assert len(tables) == 4
    assert [
        equation.metadata["equation_number"]
        for equation in equations
    ] == ["1", "2", "3", "4", "5", "6"]
    assert "Response = VLM" in equations[-1].text

    full_text = "\n".join(item.text for item in texts)
    assert "RAG-Anything" in full_text

    for table in tables:
        assert table.text.strip()
        assert "|" in table.text
        assert table.asset_path
        assert Path(table.asset_path).is_file()


def test_parser_handles_two_column_equations_and_booktabs(
    tmp_path: Path,
) -> None:
    pdf_path = Path("D:/projiect/sh/Reason-Plan-ReAct.pdf")
    parser = PyMuPDFParser(tmp_path / "assets")

    document = parser.parse(pdf_path)

    tables = [
        item for item in document.items
        if item.type == ContentType.TABLE
    ]
    equations = [
        item for item in document.items
        if item.type == ContentType.EQUATION
    ]
    images = [
        item for item in document.items
        if item.type == ContentType.IMAGE
    ]

    assert len(images) == 1
    setup = next(
        item
        for item in document.items
        if item.type == ContentType.TEXT
        and item.text.startswith("Experimental Setup")
    )
    assert setup.metadata["headings"] == [
        {"text": "Experimental Setup", "level": 1},
        {"text": "Dataset", "level": 2},
    ]
    assert len(tables) == 3
    assert all(
        table.metadata["detection_method"] == "booktabs"
        for table in tables
    )
    assert all("Coffee" in table.text for table in tables)
    assert tables[2].bbox.x1 > 500
    assert "| React | 0.63 | 0.32 | 0.65 |" in tables[2].text
    assert len(equations) == 4
    assert [equation.page_idx for equation in equations] == [3, 3, 4, 4]
    equation_texts = [equation.text for equation in equations]
    assert equation_texts[0].startswith("Sata = 1")
    assert equation_texts[1].startswith("MaxAcca = max")
    assert equation_texts[2].startswith("AverageAcca = 1")
    assert "∑" in equation_texts[2]
    assert equation_texts[3] == "CPSa = Sata · MaxAcca"


def test_parser_detects_display_equations_without_reclassifying_prose(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "equations.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "The threshold x = 3 is used in every test.",
    )
    page.insert_text(
        (72, 95),
        "The value x = 3 is the first setting (2)",
    )
    page.insert_text((220, 130), "E = mc^2")
    page.insert_text((500, 130), "(1)")
    page.insert_text((215, 190), "loss = prediction - target")
    pdf.save(pdf_path)
    pdf.close()

    document = PyMuPDFParser(tmp_path / "assets").parse(pdf_path)
    equations = [
        item for item in document.items
        if item.type == ContentType.EQUATION
    ]
    texts = [
        item.text for item in document.items
        if item.type == ContentType.TEXT
    ]

    assert len(equations) == 2
    assert equations[0].text == "E = mc^2"
    assert equations[0].metadata["equation_number"] == "1"
    assert equations[0].metadata["detection_method"] == "numbered_display"
    assert equations[1].text == "loss = prediction - target"
    assert equations[1].metadata["equation_number"] is None
    assert equations[1].metadata["detection_method"] == "math_layout"
    assert "The threshold x = 3 is used in every test." in texts
    assert "The value x = 3 is the first setting (2)" in texts
