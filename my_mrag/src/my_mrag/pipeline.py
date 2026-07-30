from __future__ import annotations

from pathlib import Path

from my_mrag.config import Settings
from my_mrag.parsers import PyMuPDFParser
from my_mrag.schemas import ParsedDocument
from my_mrag.storage import JsonDocumentStore


class IngestionPipeline:
    """Parse a source document and persist its normalized representation."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.settings.ensure_directories()
        self.store = JsonDocumentStore(self.settings.parsed_dir)
        self.pdf_parser = PyMuPDFParser(self.settings.assets_dir)

    def ingest(self, file_path: str | Path) -> tuple[ParsedDocument, Path]:
        source = Path(file_path).expanduser().resolve()
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported document type: {source.suffix}")
        document = self.pdf_parser.parse(source)
        stored_path = self.store.save(document)
        return document, stored_path

    def load(self, document_id: str) -> ParsedDocument:
        return self.store.load(document_id)

