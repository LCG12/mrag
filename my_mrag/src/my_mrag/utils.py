from __future__ import annotations

import hashlib
import re
from pathlib import Path


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{suffix}"


def safe_extension(value: str | None, default: str = "bin") -> str:
    extension = (value or default).lower().lstrip(".")
    if not re.fullmatch(r"[a-z0-9]{1,8}", extension):
        return default
    return extension

