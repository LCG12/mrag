from __future__ import annotations

from pathlib import Path
from typing import Protocol

from my_mrag.schemas import ParsedDocument


class DocumentParser(Protocol):
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a document into the normalized multimodal schema."""

