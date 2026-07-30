from __future__ import annotations

import json
from pathlib import Path

import pytest

from my_mrag.config import Settings
from my_mrag.embeddings import (
    LocalEmbeddingSettings,
    load_embedding_settings,
)


def _settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        parsed_dir=data_dir / "parsed",
        assets_dir=data_dir / "assets",
        analysis_dir=data_dir / "analysis",
        lightrag_dir=data_dir / "lightrag",
    )


def _fake_local_model(path: Path) -> None:
    path.mkdir()
    (path / "1_Pooling").mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "modules.json").write_text("[]", encoding="utf-8")
    (path / "1_Pooling" / "config.json").write_text(
        json.dumps({"word_embedding_dimension": 1024}),
        encoding="utf-8",
    )


def test_local_embedding_settings_use_model_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "qwen3-embedding-0.6b"
    _fake_local_model(model_path)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv(
        "EMBEDDING_MODEL_PATH",
        "qwen3-embedding-0.6b",
    )
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)

    config = load_embedding_settings(_settings(tmp_path))

    assert isinstance(config, LocalEmbeddingSettings)
    assert config.model_path == model_path
    assert config.dimension == 1024
    assert config.query_prompt_name == "query"
    assert config.public_dict()["provider"] == "local"


def test_local_embedding_dimension_cannot_exceed_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_local_model(tmp_path / "model")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "model")
    monkeypatch.setenv("EMBEDDING_DIM", "2048")

    with pytest.raises(ValueError, match="exceeds model dimension"):
        load_embedding_settings(_settings(tmp_path))
