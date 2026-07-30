from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from my_mrag.config import Settings


class EmbeddingModel(Protocol):
    async def __call__(
        self,
        texts: list[str],
        context: str = "document",
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass(frozen=True)
class OpenAIEmbeddingSettings:
    api_key: str
    base_url: str
    model_name: str
    dimension: int
    max_token_size: int = 8192
    provider: str = "openai"

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("api_key")
        return payload


@dataclass(frozen=True)
class LocalEmbeddingSettings:
    model_path: Path
    dimension: int
    max_token_size: int = 8192
    batch_size: int = 4
    device: str = "auto"
    query_prompt_name: str = "query"
    provider: str = "local"

    @property
    def model_name(self) -> str:
        return f"{self.model_path.name}-local-{self.dimension}"

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        payload["model_name"] = self.model_name
        return payload


EmbeddingSettings = OpenAIEmbeddingSettings | LocalEmbeddingSettings


def load_embedding_settings(settings: Settings) -> EmbeddingSettings:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider == "local":
        return _load_local_settings(settings)
    if provider == "openai":
        return _load_openai_settings()
    raise ValueError(
        "EMBEDDING_PROVIDER must be either 'local' or 'openai'"
    )


def build_embedding_model(
    config: EmbeddingSettings,
) -> EmbeddingModel:
    if isinstance(config, LocalEmbeddingSettings):
        return LocalQwen3EmbeddingModel(config)
    return OpenAIEmbeddingModel(config)


def _load_openai_settings() -> OpenAIEmbeddingSettings:
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
    model_name = os.getenv("EMBEDDING_MODEL", "").strip()
    dimension = _positive_int("EMBEDDING_DIM")
    if not api_key or not model_name:
        raise RuntimeError(
            "Configure EMBEDDING_API_KEY, EMBEDDING_MODEL, and "
            "EMBEDDING_DIM for the OpenAI embedding provider"
        )
    return OpenAIEmbeddingSettings(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        dimension=dimension,
        max_token_size=_positive_int(
            "EMBEDDING_MAX_TOKENS",
            default=8192,
        ),
    )


def _load_local_settings(settings: Settings) -> LocalEmbeddingSettings:
    configured_path = os.getenv(
        "EMBEDDING_MODEL_PATH",
        "qwen3-embedding-0.6b",
    ).strip()
    model_path = Path(configured_path).expanduser()
    if not model_path.is_absolute():
        model_path = settings.project_root / model_path
    model_path = model_path.resolve()
    required_files = [
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "modules.json",
        model_path / "1_Pooling" / "config.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Local embedding model is incomplete; missing: "
            + ", ".join(missing)
        )

    pooling_config = json.loads(
        (model_path / "1_Pooling" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    detected_dimension = int(pooling_config["word_embedding_dimension"])
    configured_dimension = os.getenv("EMBEDDING_DIM", "").strip()
    dimension = (
        int(configured_dimension)
        if configured_dimension
        else detected_dimension
    )
    if dimension > detected_dimension:
        raise ValueError(
            f"EMBEDDING_DIM={dimension} exceeds model dimension "
            f"{detected_dimension}"
        )
    if dimension <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")

    device = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            "EMBEDDING_DEVICE must be 'auto', 'cuda', or 'cpu'"
        )
    return LocalEmbeddingSettings(
        model_path=model_path,
        dimension=dimension,
        max_token_size=_positive_int(
            "EMBEDDING_MAX_TOKENS",
            default=8192,
        ),
        batch_size=_positive_int(
            "EMBEDDING_BATCH_SIZE",
            default=4,
        ),
        device=device,
        query_prompt_name=os.getenv(
            "EMBEDDING_QUERY_PROMPT_NAME",
            "query",
        ).strip(),
    )


def _positive_int(name: str, default: int | None = None) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        if default is None:
            raise RuntimeError(f"Configure {name}")
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class OpenAIEmbeddingModel:
    def __init__(self, config: OpenAIEmbeddingSettings):
        self.config = config

    async def __call__(
        self,
        texts: list[str],
        context: str = "document",
        **kwargs: Any,
    ) -> Any:
        import numpy as np
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )
        response = await client.embeddings.create(
            model=self.config.model_name,
            input=texts,
            encoding_format="float",
        )
        vectors = np.asarray(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        self._validate_shape(vectors)
        return vectors

    def _validate_shape(self, vectors: Any) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.config.dimension:
            actual = vectors.shape[1] if vectors.ndim == 2 else "invalid"
            raise RuntimeError(
                f"Embedding dimension mismatch: configured "
                f"{self.config.dimension}, received {actual}"
            )


class LocalQwen3EmbeddingModel:
    """Lazy, context-aware SentenceTransformer wrapper for Qwen3 Embedding."""

    def __init__(self, config: LocalEmbeddingSettings):
        self.config = config
        self._model: Any = None
        self._encode_lock = asyncio.Lock()
        self.resolved_device = ""

    async def __call__(
        self,
        texts: list[str],
        context: str = "document",
        **kwargs: Any,
    ) -> Any:
        if not texts:
            raise ValueError("Embedding input cannot be empty")
        if context not in {"document", "query"}:
            raise ValueError(
                "Embedding context must be 'document' or 'query'"
            )
        async with self._encode_lock:
            return await asyncio.to_thread(
                self._encode,
                texts,
                context,
            )

    def _encode(self, texts: list[str], context: str) -> Any:
        import numpy as np

        model = self._get_model()
        prompt_name = (
            self.config.query_prompt_name
            if context == "query"
            else None
        )
        vectors = model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=self.config.dimension,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape != (len(texts), self.config.dimension):
            raise RuntimeError(
                f"Local embedding shape mismatch: expected "
                f"({len(texts)}, {self.config.dimension}), "
                f"received {vectors.shape}"
            )
        return vectors

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install local embedding dependencies with: "
                "pip install -e .[local-embeddings]"
            ) from exc

        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "EMBEDDING_DEVICE=cuda but CUDA is unavailable in PyTorch"
            )

        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            model_kwargs["dtype"] = torch.float16
        model = SentenceTransformer(
            str(self.config.model_path),
            device=device,
            local_files_only=True,
            model_kwargs=model_kwargs,
            truncate_dim=self.config.dimension,
        )
        model.max_seq_length = self.config.max_token_size
        if hasattr(model, "tokenizer"):
            model.tokenizer.padding_side = "left"

        if (
            self.config.query_prompt_name
            and self.config.query_prompt_name not in model.prompts
        ):
            raise ValueError(
                f"Query prompt '{self.config.query_prompt_name}' is not "
                f"defined by the local model"
            )
        self._model = model
        self.resolved_device = device
        return model
