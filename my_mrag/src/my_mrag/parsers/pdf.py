from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from my_mrag.schemas import BoundingBox, ContentItem, ContentType, ParsedDocument
from my_mrag.utils import file_sha256, safe_extension, stable_id


_CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|table\s*\d+|表\s*\d+)",
    re.IGNORECASE,
)


@dataclass
class _PageCandidate:
    type: ContentType
    bbox: tuple[float, float, float, float]
    text: str = ""
    asset_path: str | None = None
    metadata: dict[str, Any] | None = None


class PyMuPDFParser:
    """Extract text blocks, images, and detected tables from PDF files."""

    def __init__(self, assets_root: str | Path):
        self.assets_root = Path(assets_root).resolve()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        source = Path(file_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"PDF not found: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"PyMuPDFParser only accepts PDF files: {source}")

        document_hash = file_sha256(source)
        document_id = stable_id("doc", document_hash)
        asset_dir = self.assets_root / document_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        items: list[ContentItem] = []
        with fitz.open(source) as pdf:
            for page_idx, page in enumerate(pdf):
                page_candidates = self._parse_page(
                    page=page,
                    page_idx=page_idx,
                    document_id=document_id,
                    asset_dir=asset_dir,
                )
                for candidate in page_candidates:
                    order_idx = len(items)
                    item_id = stable_id(
                        "item",
                        document_id,
                        page_idx,
                        order_idx,
                        candidate.type.value,
                        candidate.text,
                        candidate.asset_path,
                    )
                    items.append(
                        ContentItem(
                            item_id=item_id,
                            document_id=document_id,
                            type=candidate.type,
                            page_idx=page_idx,
                            order_idx=order_idx,
                            text=candidate.text,
                            bbox=BoundingBox.from_value(candidate.bbox),
                            asset_path=candidate.asset_path,
                            metadata=candidate.metadata or {},
                        )
                    )

            self._attach_nearby_captions(items)
            metadata = {
                key: value
                for key, value in (pdf.metadata or {}).items()
                if value not in (None, "")
            }
            page_count = pdf.page_count

        return ParsedDocument(
            document_id=document_id,
            source_path=str(source),
            parser="pymupdf",
            page_count=page_count,
            items=items,
            metadata=metadata,
        )

    def _parse_page(
        self,
        page: fitz.Page,
        page_idx: int,
        document_id: str,
        asset_dir: Path,
    ) -> list[_PageCandidate]:
        table_candidates = self._extract_tables(
            page, page_idx, document_id, asset_dir
        )
        table_boxes = [candidate.bbox for candidate in table_candidates]
        candidates = list(table_candidates)

        page_dict = page.get_text("dict", sort=True)
        for block_idx, block in enumerate(page_dict.get("blocks", [])):
            bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
            block_type = int(block.get("type", -1))

            if block_type == 0:
                text = self._text_from_block(block)
                if (
                    not text
                    or not self._is_horizontal_text_block(block)
                    or self._mostly_inside_any(bbox, table_boxes)
                ):
                    continue
                candidates.append(
                    _PageCandidate(
                        type=ContentType.TEXT,
                        bbox=bbox,
                        text=text,
                        metadata={"source_block_idx": block_idx},
                    )
                )
            elif block_type == 1:
                image_bytes = block.get("image")
                if not image_bytes:
                    continue
                width = int(block.get("width") or 0)
                height = int(block.get("height") or 0)
                display_width = max(bbox[2] - bbox[0], 0)
                display_height = max(bbox[3] - bbox[1], 0)
                if display_width < 16 or display_height < 16:
                    continue
                extension = safe_extension(block.get("ext"), default="png")
                image_id = stable_id(
                    "image", document_id, page_idx, block_idx, len(image_bytes)
                )
                image_path = asset_dir / f"{image_id}.{extension}"
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)
                candidates.append(
                    _PageCandidate(
                        type=ContentType.IMAGE,
                        bbox=bbox,
                        asset_path=str(image_path.resolve()),
                        metadata={
                            "source_block_idx": block_idx,
                            "width": width,
                            "height": height,
                            "display_width": round(display_width, 3),
                            "display_height": round(display_height, 3),
                            "extension": extension,
                        },
                    )
                )

        return self._sort_reading_order(candidates, page.rect)

    def _extract_tables(
        self,
        page: fitz.Page,
        page_idx: int,
        document_id: str,
        asset_dir: Path,
    ) -> list[_PageCandidate]:
        try:
            tables = page.find_tables().tables
        except Exception:
            return []

        candidates: list[_PageCandidate] = []
        for table_idx, table in enumerate(tables):
            bbox = tuple(float(value) for value in table.bbox)
            markdown = self._table_to_markdown(table)
            if not markdown.strip():
                continue
            table_id = stable_id("table", document_id, page_idx, table_idx, markdown)
            image_path = asset_dir / f"{table_id}.png"
            if not image_path.exists():
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    clip=fitz.Rect(bbox),
                    alpha=False,
                )
                pixmap.save(image_path)
            candidates.append(
                _PageCandidate(
                    type=ContentType.TABLE,
                    bbox=bbox,
                    text=markdown,
                    asset_path=str(image_path.resolve()),
                    metadata={"table_index": table_idx},
                )
            )
        return candidates

    @staticmethod
    def _table_to_markdown(table: Any) -> str:
        try:
            return str(table.to_markdown()).strip()
        except Exception:
            rows = table.extract() or []
            if not rows:
                return ""
            normalized = [
                [str(cell or "").replace("\n", " ").strip() for cell in row]
                for row in rows
            ]
            header = normalized[0]
            separator = ["---"] * len(header)
            body = normalized[1:]
            return "\n".join(
                "| " + " | ".join(row) + " |"
                for row in [header, separator, *body]
            )

    @staticmethod
    def _text_from_block(block: dict[str, Any]) -> str:
        lines: list[str] = []
        for line in block.get("lines", []):
            text = "".join(
                str(span.get("text", "")) for span in line.get("spans", [])
            ).strip()
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    @staticmethod
    def _is_horizontal_text_block(block: dict[str, Any]) -> bool:
        directions = [
            tuple(line.get("dir", (1.0, 0.0)))
            for line in block.get("lines", [])
        ]
        if not directions:
            return True
        horizontal_lines = sum(
            1
            for x_direction, y_direction in directions
            if abs(float(x_direction)) >= 0.9 and abs(float(y_direction)) <= 0.1
        )
        return horizontal_lines >= max(1, len(directions) // 2)

    @staticmethod
    def _sort_reading_order(
        candidates: list[_PageCandidate],
        page_rect: fitz.Rect,
    ) -> list[_PageCandidate]:
        """Order full-width blocks and two-column content for research papers."""
        if not candidates:
            return []

        center_x = (page_rect.x0 + page_rect.x1) / 2
        spanning: list[_PageCandidate] = []
        left: list[_PageCandidate] = []
        right: list[_PageCandidate] = []

        for candidate in candidates:
            x0, _, x1, _ = candidate.bbox
            if x0 < center_x < x1:
                spanning.append(candidate)
            elif (x0 + x1) / 2 < center_x:
                left.append(candidate)
            else:
                right.append(candidate)

        position_key = lambda item: (
            round(item.bbox[1], 1),
            round(item.bbox[0], 1),
            item.type.value,
        )
        spanning.sort(key=position_key)
        left.sort(key=position_key)
        right.sort(key=position_key)

        ordered: list[_PageCandidate] = []
        left_index = 0
        right_index = 0
        tolerance = 2.0

        for full_width_item in spanning:
            boundary = full_width_item.bbox[1] + tolerance
            left_before: list[_PageCandidate] = []
            while (
                left_index < len(left)
                and left[left_index].bbox[3] <= boundary
            ):
                left_before.append(left[left_index])
                left_index += 1
            right_before: list[_PageCandidate] = []
            while (
                right_index < len(right)
                and right[right_index].bbox[3] <= boundary
            ):
                right_before.append(right[right_index])
                right_index += 1
            ordered.extend(left_before)
            ordered.extend(right_before)
            ordered.append(full_width_item)

        ordered.extend(left[left_index:])
        ordered.extend(right[right_index:])
        return ordered

    @staticmethod
    def _mostly_inside_any(
        bbox: tuple[float, float, float, float],
        containers: list[tuple[float, float, float, float]],
    ) -> bool:
        block = fitz.Rect(bbox)
        block_area = max(block.get_area(), 1.0)
        for container_bbox in containers:
            intersection = block & fitz.Rect(container_bbox)
            if not intersection.is_empty and intersection.get_area() / block_area >= 0.6:
                return True
        return False

    @staticmethod
    def _attach_nearby_captions(items: list[ContentItem]) -> None:
        text_items = [item for item in items if item.type == ContentType.TEXT]
        for item in items:
            if item.type not in (ContentType.IMAGE, ContentType.TABLE) or not item.bbox:
                continue
            matches: list[tuple[float, ContentItem]] = []
            for text_item in text_items:
                if text_item.page_idx != item.page_idx or not text_item.bbox:
                    continue
                if not _CAPTION_RE.match(text_item.text):
                    continue
                vertical_distance = min(
                    abs(text_item.bbox.y0 - item.bbox.y1),
                    abs(item.bbox.y0 - text_item.bbox.y1),
                )
                if vertical_distance <= 140:
                    matches.append((vertical_distance, text_item))
            if matches:
                matches.sort(key=lambda pair: pair[0])
                item.captions = [matches[0][1].text]
