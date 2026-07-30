from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the ingestion pipeline."""

    project_root: Path
    data_dir: Path
    parsed_dir: Path
    assets_dir: Path
    analysis_dir: Path

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> "Settings":
        root = PROJECT_ROOT
        configured_data_dir = data_dir or os.getenv("MY_MRAG_DATA_DIR", "data")
        data_path = Path(configured_data_dir)
        if not data_path.is_absolute():
            data_path = root / data_path
        data_path = data_path.resolve()
        return cls(
            project_root=root,
            data_dir=data_path,
            parsed_dir=data_path / "parsed",
            assets_dir=data_path / "assets",
            analysis_dir=data_path / "analysis",
        )

    def ensure_directories(self) -> None:
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
