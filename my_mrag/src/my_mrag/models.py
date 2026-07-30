from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Protocol

from my_mrag.schemas import AnalysisRequest


class AnalysisModel(Protocol):
    model_name: str

    async def complete(self, request: AnalysisRequest) -> str:
        ...


class OpenAICompatibleModel:
    """Small adapter for text or vision OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        max_tokens: int = 1600,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        if not model_name:
            raise ValueError("model_name is required")
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        required: bool = True,
    ) -> "OpenAICompatibleModel | None":
        api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
        base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
        model_name = os.getenv(f"{prefix}_MODEL", "").strip()
        if not api_key or not model_name:
            if required:
                raise RuntimeError(
                    f"{prefix}_API_KEY and {prefix}_MODEL must be configured"
                )
            return None
        return cls(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

    async def complete(self, request: AnalysisRequest) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the model dependency with: pip install -e .[models]"
            ) from exc

        content: str | list[dict[str, object]]
        if request.image_paths:
            blocks: list[dict[str, object]] = [
                {"type": "text", "text": request.prompt}
            ]
            for image_path in request.image_paths:
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_data_url(Path(image_path))
                        },
                    }
                )
            content = blocks
        else:
            content = request.prompt

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _image_data_url(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
