from __future__ import annotations

import json
from pathlib import Path

from my_mrag.schemas import AnalysisRequest, ModalAnalysis, ParsedDocument


class JsonDocumentStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, document: ParsedDocument) -> Path:
        target = self.directory / f"{document.document_id}.json"
        temp_target = target.with_suffix(".json.tmp")
        with temp_target.open("w", encoding="utf-8") as stream:
            json.dump(document.to_dict(), stream, ensure_ascii=False, indent=2)
        temp_target.replace(target)
        return target

    def load(self, document_id: str) -> ParsedDocument:
        path = self.directory / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Parsed document not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            return ParsedDocument.from_dict(json.load(stream))


class JsonAnalysisStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_requests(
        self,
        document_id: str,
        requests: list[AnalysisRequest],
    ) -> Path:
        return self._save(
            self.directory / f"{document_id}.requests.json",
            {
                "document_id": document_id,
                "requests": [request.to_dict() for request in requests],
            },
        )

    def save_analyses(
        self,
        document_id: str,
        analyses: list[ModalAnalysis],
    ) -> Path:
        return self._save(
            self.directory / f"{document_id}.json",
            {
                "document_id": document_id,
                "analyses": [analysis.to_dict() for analysis in analyses],
            },
        )

    def load_analyses(self, document_id: str) -> list[ModalAnalysis]:
        path = self.directory / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Multimodal analysis not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return [
            ModalAnalysis.from_dict(item)
            for item in payload.get("analyses", [])
        ]

    @staticmethod
    def _save(path: Path, payload: dict[str, object]) -> Path:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        temp_path.replace(path)
        return path
