from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from my_mrag.schemas import BoundingBox, ContentItem, ContentType, ParsedDocument
from my_mrag.utils import file_sha256, stable_id


_CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|table\s*\d+|表\s*\d+)",
    re.IGNORECASE,
)


_FIGURE_CAPTION_RE = re.compile(
    r"^\s*fig(?:ure)?\.?\s*\d+",
    re.IGNORECASE,
)


_TABLE_CAPTION_RE = re.compile(
    r"^\s*table\s+\d+\b",
    re.IGNORECASE | re.MULTILINE,
)


_EQUATION_NUMBER_RE = re.compile(
    r"(?:^|\s)\(\s*(\d{1,3})\s*\)\s*$",
)


_MATH_FONT_MARKERS = (
    "math",
    "symbol",
    "cmmi",
    "cmsy",
    "cmex",
    "msam",
    "msbm",
    "stix",
)


_MATH_SYMBOLS = frozenset(
    "=+×÷∈∉∪∩⊂⊆⊇≤≥≈≠→←↔∑∏∫√∥⋆∞∂∇{}[]^_"
)


_MATH_OPERATORS = frozenset("=+-<>^_×÷−·∑∏∫√≈≠≤≥")


@dataclass
class _PageCandidate:
    type: ContentType
    bbox: tuple[float, float, float, float]
    text: str = ""
    asset_path: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _EmbeddedImage:
    bbox: tuple[float, float, float, float]
    source_block_idx: int
    width: int
    height: int


@dataclass(frozen=True)
class _TextFragment:
    bbox: tuple[float, float, float, float]
    text: str
    source_block_idx: int
    source_line_idx: int
    math_character_count: int
    character_count: int

    @property
    def math_ratio(self) -> float:
        return self.math_character_count / max(self.character_count, 1)


@dataclass(frozen=True)
class _TextHeading:
    text: str
    level: int


@dataclass(frozen=True)
class _TextBlock:
    bbox: tuple[float, float, float, float]
    text: str
    source_block_idx: int
    math_character_count: int
    character_count: int
    fragments: tuple[_TextFragment, ...] = ()
    headings: tuple[_TextHeading, ...] = ()

    @property
    def math_ratio(self) -> float:
        return self.math_character_count / max(self.character_count, 1)


class PyMuPDFParser:
    """Extract text, equations, images, and detected tables from PDF files."""

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
        pending_equations: list[tuple[int, _PageCandidate]] = []
        with fitz.open(source) as pdf:
            for page_idx, page in enumerate(pdf):
                page_candidates = self._parse_page(
                    page=page,
                    page_idx=page_idx,
                    document_id=document_id,
                    asset_dir=asset_dir,
                )
                for candidate in page_candidates:
                    if candidate.type == ContentType.EQUATION:
                        pending_equations.append((page_idx, candidate))
                        continue
                    self._append_candidate(
                        items,
                        document_id,
                        page_idx,
                        candidate,
                    )

            for page_idx, candidate in pending_equations:
                self._append_candidate(
                    items,
                    document_id,
                    page_idx,
                    candidate,
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

    @staticmethod
    def _append_candidate(
        items: list[ContentItem],
        document_id: str,
        page_idx: int,
        candidate: _PageCandidate,
    ) -> None:
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
        embedded_images: list[_EmbeddedImage] = []
        text_blocks: list[_TextBlock] = []

        page_dict = page.get_text("dict", sort=True)
        for block_idx, block in enumerate(page_dict.get("blocks", [])):
            bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
            block_type = int(block.get("type", -1))

            if block_type == 0:
                text_block = self._text_block(block, block_idx)
                if (
                    not text_block.text
                    or not self._is_horizontal_text_block(block)
                    or self._mostly_inside_any(bbox, table_boxes)
                ):
                    continue
                text_blocks.append(text_block)
            elif block_type == 1:
                image_bytes = block.get("image")
                if not image_bytes:
                    continue
                width = int(block.get("width") or 0)
                height = int(block.get("height") or 0)
                display_width = max(bbox[2] - bbox[0], 0)
                display_height = max(bbox[3] - bbox[1], 0)
                if display_width < 2 or display_height < 2:
                    continue
                embedded_images.append(
                    _EmbeddedImage(
                        bbox=bbox,
                        source_block_idx=block_idx,
                        width=width,
                        height=height,
                    )
                )

        equation_candidates, _ = self._extract_equations(
            text_blocks,
            page.rect,
        )
        candidates.extend(
            _PageCandidate(
                type=ContentType.TEXT,
                bbox=block.bbox,
                text=block.text,
                metadata={
                    "source_block_idx": block.source_block_idx,
                    **(
                        {
                            "headings": [
                                {
                                    "text": heading.text,
                                    "level": heading.level,
                                }
                                for heading in block.headings
                            ]
                        }
                        if block.headings
                        else {}
                    ),
                },
            )
            for block in text_blocks
        )

        figure_captions = [
            candidate
            for candidate in candidates
            if candidate.type == ContentType.TEXT
            and _FIGURE_CAPTION_RE.match(candidate.text)
        ]
        candidates.extend(
            self._extract_image_groups(
                page=page,
                page_idx=page_idx,
                document_id=document_id,
                asset_dir=asset_dir,
                images=embedded_images,
                figure_captions=figure_captions,
                table_boxes=table_boxes,
            )
        )

        return [
            *self._sort_reading_order(candidates, page.rect),
            *self._sort_reading_order(equation_candidates, page.rect),
        ]

    def _extract_image_groups(
        self,
        page: fitz.Page,
        page_idx: int,
        document_id: str,
        asset_dir: Path,
        images: list[_EmbeddedImage],
        figure_captions: list[_PageCandidate],
        table_boxes: list[tuple[float, float, float, float]],
    ) -> list[_PageCandidate]:
        candidates: list[_PageCandidate] = []
        for group_idx, group in enumerate(self._group_images(images)):
            bbox = self._union_image_boxes(group)
            if self._mostly_inside_any(bbox, table_boxes):
                continue

            caption = self._nearest_caption_below(bbox, figure_captions)
            previous_caption = self._nearest_caption_above(bbox, figure_captions)
            clip = fitz.Rect(bbox)
            if len(group) > 1 and caption is not None:
                clip.x0 = min(clip.x0, caption.bbox[0])
                clip.x1 = max(clip.x1, caption.bbox[2])

            padding = 4.0
            padded_y0 = clip.y0 - padding
            if previous_caption is not None:
                padded_y0 = max(padded_y0, previous_caption.bbox[3] + 1)
            clip = fitz.Rect(
                max(page.rect.x0, clip.x0 - padding),
                max(page.rect.y0, padded_y0),
                min(page.rect.x1, clip.x1 + padding),
                min(page.rect.y1, clip.y1 + padding),
            )
            image_id = stable_id(
                "image",
                document_id,
                page_idx,
                group_idx,
                tuple(round(value, 3) for value in clip),
                tuple(image.source_block_idx for image in group),
            )
            image_path = asset_dir / f"{image_id}.png"
            if not image_path.exists():
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    clip=clip,
                    alpha=False,
                    annots=False,
                )
                pixmap.save(image_path)
            else:
                pixmap = fitz.Pixmap(image_path)

            candidates.append(
                _PageCandidate(
                    type=ContentType.IMAGE,
                    bbox=tuple(float(value) for value in clip),
                    asset_path=str(image_path.resolve()),
                    metadata={
                        "extraction_method": "page_region",
                        "source_block_indexes": [
                            image.source_block_idx for image in group
                        ],
                        "embedded_image_count": len(group),
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "display_width": round(clip.width, 3),
                        "display_height": round(clip.height, 3),
                        "extension": "png",
                    },
                )
            )
        return candidates

    @staticmethod
    def _group_images(
        images: list[_EmbeddedImage],
        vertical_gap: float = 12.0,
    ) -> list[list[_EmbeddedImage]]:
        """Group image layers occupying the same vertical figure region."""
        groups: list[list[_EmbeddedImage]] = []
        for image in sorted(images, key=lambda item: (item.bbox[1], item.bbox[0])):
            if not groups:
                groups.append([image])
                continue
            current_bottom = max(item.bbox[3] for item in groups[-1])
            if image.bbox[1] <= current_bottom + vertical_gap:
                groups[-1].append(image)
            else:
                groups.append([image])
        return groups

    @staticmethod
    def _union_image_boxes(
        images: list[_EmbeddedImage],
    ) -> tuple[float, float, float, float]:
        return (
            min(image.bbox[0] for image in images),
            min(image.bbox[1] for image in images),
            max(image.bbox[2] for image in images),
            max(image.bbox[3] for image in images),
        )

    @staticmethod
    def _nearest_caption_below(
        bbox: tuple[float, float, float, float],
        captions: list[_PageCandidate],
        max_distance: float = 180.0,
    ) -> _PageCandidate | None:
        matches = [
            caption
            for caption in captions
            if caption.bbox[1] >= bbox[3] - 2
            and caption.bbox[1] - bbox[3] <= max_distance
        ]
        if not matches:
            return None
        return min(matches, key=lambda caption: caption.bbox[1] - bbox[3])

    @staticmethod
    def _nearest_caption_above(
        bbox: tuple[float, float, float, float],
        captions: list[_PageCandidate],
        max_distance: float = 40.0,
    ) -> _PageCandidate | None:
        matches = [
            caption
            for caption in captions
            if caption.bbox[3] <= bbox[1] + 2
            and bbox[1] - caption.bbox[3] <= max_distance
        ]
        if not matches:
            return None
        return min(matches, key=lambda caption: bbox[1] - caption.bbox[3])

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

        detection_method = "lines"
        if not tables and _TABLE_CAPTION_RE.search(page.get_text("text")):
            try:
                tables = page.find_tables(
                    vertical_strategy="text",
                    horizontal_strategy="lines",
                ).tables
                detection_method = "booktabs"
            except Exception:
                return []

        candidates: list[_PageCandidate] = []
        for table_idx, table in enumerate(tables):
            bbox = tuple(float(value) for value in table.bbox)
            if (
                detection_method == "booktabs"
                and bbox[3] - bbox[1] > page.rect.height * 0.55
            ):
                continue
            if detection_method == "booktabs":
                bbox = self._expand_booktabs_bbox(page, bbox)
                markdown = self._booktabs_to_markdown(page, bbox)
            else:
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
                    metadata={
                        "table_index": table_idx,
                        "detection_method": detection_method,
                    },
                )
            )
        return candidates

    @staticmethod
    def _expand_booktabs_bbox(
        page: fitz.Page,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        minimum_width = (x1 - x0) * 0.75
        horizontal_lines: list[tuple[float, float, float]] = []
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item[0] != "l":
                    continue
                start, end = item[1], item[2]
                if abs(float(start.y) - float(end.y)) > 1.0:
                    continue
                line_x0 = min(float(start.x), float(end.x))
                line_x1 = max(float(start.x), float(end.x))
                line_y = (float(start.y) + float(end.y)) / 2
                overlap = min(x1, line_x1) - max(x0, line_x0)
                if (
                    line_x1 - line_x0 >= minimum_width
                    and overlap > 0
                    and y0 - 20.0 <= line_y <= y1 + 3.0
                ):
                    horizontal_lines.append((line_x0, line_y, line_x1))

        if not horizontal_lines:
            return bbox
        return (
            min(x0, *(line[0] for line in horizontal_lines)),
            min(y0, *(line[1] for line in horizontal_lines)),
            max(x1, *(line[2] for line in horizontal_lines)),
            max(y1, *(line[1] for line in horizontal_lines)),
        )

    @staticmethod
    def _booktabs_to_markdown(
        page: fitz.Page,
        bbox: tuple[float, float, float, float],
    ) -> str:
        fragments: list[tuple[float, float, str]] = []
        for block in page.get_text("dict", sort=True).get("blocks", []):
            if int(block.get("type", -1)) != 0:
                continue
            for line in block.get("lines", []):
                line_bbox = tuple(
                    float(value) for value in line.get("bbox", (0, 0, 0, 0))
                )
                center_x = (line_bbox[0] + line_bbox[2]) / 2
                center_y = (line_bbox[1] + line_bbox[3]) / 2
                if not (
                    bbox[0] - 1 <= center_x <= bbox[2] + 1
                    and bbox[1] - 1 <= center_y <= bbox[3] + 1
                ):
                    continue
                text = "".join(
                    str(span.get("text", ""))
                    for span in line.get("spans", [])
                ).strip()
                if text:
                    fragments.append((center_y, line_bbox[0], text))

        rows: list[list[tuple[float, float, str]]] = []
        for fragment in sorted(fragments, key=lambda value: (value[0], value[1])):
            if not rows:
                rows.append([fragment])
                continue
            row_center = sum(item[0] for item in rows[-1]) / len(rows[-1])
            if abs(fragment[0] - row_center) <= 2.0:
                rows[-1].append(fragment)
            else:
                rows.append([fragment])

        cells = [
            [
                item[2].replace("|", r"\|")
                for item in sorted(row, key=lambda value: value[1])
            ]
            for row in rows
        ]
        if not cells:
            return ""
        column_count = max(len(row) for row in cells)
        normalized = [
            [*row, *([""] * (column_count - len(row)))]
            for row in cells
        ]
        separator = ["---"] * column_count
        return "\n".join(
            "| " + " | ".join(row) + " |"
            for row in [normalized[0], separator, *normalized[1:]]
        )

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

    @classmethod
    def _text_from_block(cls, block: dict[str, Any]) -> str:
        lines: list[str] = []
        for line in block.get("lines", []):
            text = "".join(
                cls._text_from_span(span) for span in line.get("spans", [])
            ).strip()
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    @staticmethod
    def _text_from_span(span: dict[str, Any]) -> str:
        text = str(span.get("text", ""))
        font = str(span.get("font", "")).casefold()
        if "cmex" in font and text == "X":
            return "∑"
        return text

    @classmethod
    def _text_block(
        cls,
        block: dict[str, Any],
        block_idx: int,
    ) -> _TextBlock:
        math_characters = 0
        characters = 0
        fragments: list[_TextFragment] = []
        for line_idx, line in enumerate(block.get("lines", [])):
            line_math_characters = 0
            line_characters = 0
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = cls._text_from_span(span)
                line_parts.append(span_text)
                visible_count = sum(
                    1 for character in span_text if not character.isspace()
                )
                characters += visible_count
                line_characters += visible_count
                font = str(span.get("font", "")).casefold()
                if any(marker in font for marker in _MATH_FONT_MARKERS):
                    math_characters += visible_count
                    line_math_characters += visible_count

            line_text = "".join(line_parts).strip()
            if line_text:
                fragments.append(
                    _TextFragment(
                        bbox=tuple(
                            float(value)
                            for value in line.get("bbox", block.get("bbox"))
                        ),
                        text=line_text,
                        source_block_idx=block_idx,
                        source_line_idx=line_idx,
                        math_character_count=line_math_characters,
                        character_count=line_characters,
                    )
                )

        return _TextBlock(
            bbox=tuple(
                float(value) for value in block.get("bbox", (0, 0, 0, 0))
            ),
            text=cls._text_from_block(block),
            source_block_idx=block_idx,
            math_character_count=math_characters,
            character_count=characters,
            fragments=tuple(fragments),
            headings=cls._heading_prefixes(block),
        )

    @classmethod
    def _heading_prefixes(
        cls,
        block: dict[str, Any],
    ) -> tuple[_TextHeading, ...]:
        prefix_lines: list[tuple[str, float]] = []
        for line in block.get("lines", []):
            all_spans = list(line.get("spans", []))
            spans = [
                span
                for span in all_spans
                if str(span.get("text", "")).strip()
            ]
            if not spans or not all(cls._is_bold_span(span) for span in spans):
                break
            text = "".join(
                cls._text_from_span(span) for span in all_spans
            ).strip()
            size = max(float(span.get("size") or 0) for span in spans)
            if text:
                prefix_lines.append((text, size))

        headings: list[_TextHeading] = []
        for text, size in prefix_lines:
            if size > 13.0:
                return ()
            level = cls._typographic_heading_level(text, size)
            if headings and headings[-1].level == level:
                previous = headings[-1]
                headings[-1] = _TextHeading(
                    text=f"{previous.text} {text}".strip(),
                    level=level,
                )
            else:
                headings.append(_TextHeading(text=text, level=level))
        return tuple(headings)

    @staticmethod
    def _is_bold_span(span: dict[str, Any]) -> bool:
        font = str(span.get("font", "")).casefold()
        flags = int(span.get("flags") or 0)
        return bool(flags & 16) or any(
            marker in font for marker in ("bold", "demi", "medi")
        )

    @staticmethod
    def _typographic_heading_level(text: str, size: float) -> int:
        if text.strip().upper() in {
            "ABSTRACT",
            "ACKNOWLEDGMENTS",
            "APPENDIX",
            "CONCLUSION",
            "CONCLUSIONS",
            "REFERENCES",
        }:
            return 1
        if size >= 11.5:
            return 1
        if size >= 10.5:
            return 2
        return 3

    @classmethod
    def _extract_equations(
        cls,
        blocks: list[_TextBlock],
        page_rect: fitz.Rect,
    ) -> tuple[list[_PageCandidate], set[int]]:
        """Detect numbered displays and conservative unnumbered math blocks."""
        anchors = [
            (block, match.group(1))
            for block in blocks
            if (match := _EQUATION_NUMBER_RE.search(block.text)) is not None
        ]
        assignments: dict[int, list[_TextBlock]] = {
            anchor.source_block_idx: [anchor]
            for anchor, _ in anchors
        }

        for block in blocks:
            if block.source_block_idx in assignments:
                continue
            if not cls._is_equation_fragment(block):
                continue
            compatible = [
                anchor
                for anchor, _ in anchors
                if cls._vertical_center_distance(block, anchor) <= 22.0
            ]
            if not compatible:
                continue
            anchor = min(
                compatible,
                key=lambda value: cls._vertical_center_distance(block, value),
            )
            assignments[anchor.source_block_idx].append(block)

        equations: list[_PageCandidate] = []
        consumed: set[int] = set()
        for anchor, number in anchors:
            group = assignments[anchor.source_block_idx]
            anchor_text = cls._normalize_equation_fragment(
                _EQUATION_NUMBER_RE.sub("", anchor.text)
            )
            if anchor_text:
                if not cls._looks_like_equation_text(
                    anchor_text,
                    anchor.math_ratio,
                ):
                    continue
            elif len(group) == 1:
                continue
            equation_text = cls._equation_text(group, number)
            if not equation_text or not cls._group_has_math(group, equation_text):
                continue
            source_indexes = sorted(block.source_block_idx for block in group)
            equations.append(
                _PageCandidate(
                    type=ContentType.EQUATION,
                    bbox=cls._union_text_boxes(group),
                    text=equation_text,
                    metadata={
                        "equation_number": number,
                        "equation_format": "pdf_text",
                        "detection_method": "numbered_display",
                        "source_block_indexes": source_indexes,
                    },
                )
            )
            consumed.update(source_indexes)

        fragments = [
            fragment
            for block in blocks
            if block.source_block_idx not in consumed
            for fragment in block.fragments
            if cls._is_unnumbered_display_equation(fragment, page_rect)
        ]
        for group in cls._group_unnumbered_fragments(fragments, page_rect):
            equation_text = cls._equation_text(group, None)
            if not cls._looks_like_complete_equation(group, equation_text):
                continue
            source_indexes = sorted(
                {fragment.source_block_idx for fragment in group}
            )
            equations.append(
                _PageCandidate(
                    type=ContentType.EQUATION,
                    bbox=cls._union_text_boxes(group),
                    text=equation_text,
                    metadata={
                        "equation_number": None,
                        "equation_format": "pdf_text",
                        "detection_method": "math_layout",
                        "source_block_indexes": source_indexes,
                        "source_line_indexes": [
                            {
                                "block": fragment.source_block_idx,
                                "line": fragment.source_line_idx,
                            }
                            for fragment in group
                        ],
                    },
                )
            )
            consumed.update(source_indexes)

        return equations, consumed

    @classmethod
    def _is_equation_fragment(
        cls,
        block: _TextBlock | _TextFragment,
    ) -> bool:
        text = cls._normalize_equation_fragment(block.text)
        return cls._looks_like_equation_text(text, block.math_ratio)

    @staticmethod
    def _looks_like_equation_text(text: str, math_ratio: float) -> bool:
        if not text or len(text) > 240:
            return False
        if len(text) <= 4 and re.fullmatch(r"\d+(?:\.\d+)?", text):
            return True
        symbol_count = sum(character in _MATH_OPERATORS for character in text)
        word_count = len(re.findall(r"[A-Za-z]{2,}", text))
        if math_ratio >= 0.2:
            return True
        if symbol_count >= 1 and word_count <= 4:
            return True
        return len(text) <= 12 and bool(re.search(r"[\[\]{}()]", text))

    @classmethod
    def _is_unnumbered_display_equation(
        cls,
        block: _TextFragment,
        page_rect: fitz.Rect,
    ) -> bool:
        if _EQUATION_NUMBER_RE.search(block.text):
            return False
        if not cls._is_equation_fragment(block):
            return False
        text = cls._normalize_equation_fragment(block.text)
        block_center = (block.bbox[0] + block.bbox[2]) / 2
        column = cls._column_index(block.bbox, page_rect)
        if column == 0:
            target_center = (page_rect.x0 + page_rect.x1) / 2
            tolerance = page_rect.width * 0.2
        elif column < 0:
            target_center = page_rect.x0 + page_rect.width * 0.25
            tolerance = page_rect.width * 0.15
        else:
            target_center = page_rect.x0 + page_rect.width * 0.75
            tolerance = page_rect.width * 0.15
        centered = abs(block_center - target_center) <= tolerance
        narrow = block.bbox[2] - block.bbox[0] <= page_rect.width * 0.8
        word_count = len(re.findall(r"[A-Za-z]{2,}", text))
        return centered and narrow and word_count <= 4

    @classmethod
    def _group_unnumbered_fragments(
        cls,
        fragments: list[_TextFragment],
        page_rect: fitz.Rect,
    ) -> list[list[_TextFragment]]:
        groups: list[list[_TextFragment]] = []
        ordered = sorted(
            fragments,
            key=lambda value: (value.bbox[1], value.bbox[0]),
        )
        for fragment in ordered:
            compatible: list[list[_TextFragment]] = []
            fragment_column = cls._column_index(fragment.bbox, page_rect)
            for group in groups:
                group_column = cls._column_index(
                    cls._union_text_boxes(group),
                    page_rect,
                )
                bottom = max(item.bbox[3] for item in group)
                center_distance = min(
                    cls._vertical_center_distance(fragment, item)
                    for item in group
                )
                vertical_gap = fragment.bbox[1] - bottom
                if (
                    fragment_column == group_column
                    and vertical_gap <= 7.0
                    and center_distance <= 20.0
                ):
                    compatible.append(group)
            if not compatible:
                groups.append([fragment])
                continue
            nearest = min(
                compatible,
                key=lambda group: min(
                    cls._vertical_center_distance(fragment, item)
                    for item in group
                ),
            )
            nearest.append(fragment)
        return groups

    @staticmethod
    def _looks_like_complete_equation(
        fragments: list[_TextFragment],
        equation_text: str,
    ) -> bool:
        if not any(
            character in _MATH_OPERATORS for character in equation_text
        ):
            return False
        math_characters = sum(
            fragment.math_character_count for fragment in fragments
        )
        characters = sum(fragment.character_count for fragment in fragments)
        if math_characters / max(characters, 1) >= 0.2:
            return True
        return bool(
            re.match(
                r"^[A-Za-z][A-Za-z0-9_ ]{0,40}\s*=",
                equation_text,
            )
        )

    @staticmethod
    def _column_index(
        bbox: tuple[float, float, float, float],
        page_rect: fitz.Rect,
    ) -> int:
        page_center = (page_rect.x0 + page_rect.x1) / 2
        if bbox[0] < page_center < bbox[2]:
            return 0
        return -1 if (bbox[0] + bbox[2]) / 2 < page_center else 1

    @classmethod
    def _group_has_math(
        cls,
        blocks: list[_TextBlock | _TextFragment],
        equation_text: str,
    ) -> bool:
        symbol_count = sum(
            character in _MATH_SYMBOLS for character in equation_text
        )
        math_characters = sum(block.math_character_count for block in blocks)
        return symbol_count > 0 or math_characters > 0

    @classmethod
    def _equation_text(
        cls,
        blocks: list[_TextBlock | _TextFragment],
        number: str | None,
    ) -> str:
        ordered_rows: list[list[_TextBlock | _TextFragment]] = []
        for block in sorted(
            blocks,
            key=lambda value: (
                (value.bbox[1] + value.bbox[3]) / 2,
                value.bbox[0],
            ),
        ):
            center_y = (block.bbox[1] + block.bbox[3]) / 2
            if not ordered_rows:
                ordered_rows.append([block])
                continue
            previous_centers = [
                (item.bbox[1] + item.bbox[3]) / 2
                for item in ordered_rows[-1]
            ]
            row_center = sum(previous_centers) / len(previous_centers)
            if abs(center_y - row_center) <= 6.0:
                ordered_rows[-1].append(block)
            else:
                ordered_rows.append([block])

        parts: list[str] = []
        for row in ordered_rows:
            for block in sorted(row, key=lambda value: value.bbox[0]):
                text = _EQUATION_NUMBER_RE.sub("", block.text).strip()
                normalized = cls._normalize_equation_fragment(text)
                if normalized:
                    parts.append(normalized)
        return " ".join(parts).strip()

    @staticmethod
    def _normalize_equation_fragment(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _vertical_center_distance(
        first: _TextBlock | _TextFragment,
        second: _TextBlock | _TextFragment,
    ) -> float:
        first_center = (first.bbox[1] + first.bbox[3]) / 2
        second_center = (second.bbox[1] + second.bbox[3]) / 2
        return abs(first_center - second_center)

    @staticmethod
    def _union_text_boxes(
        blocks: list[_TextBlock | _TextFragment],
    ) -> tuple[float, float, float, float]:
        return (
            min(block.bbox[0] for block in blocks),
            min(block.bbox[1] for block in blocks),
            max(block.bbox[2] for block in blocks),
            max(block.bbox[3] for block in blocks),
        )

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
            above: list[tuple[float, ContentItem]] = []
            below: list[tuple[float, ContentItem]] = []
            for text_item in text_items:
                if text_item.page_idx != item.page_idx or not text_item.bbox:
                    continue
                if not _CAPTION_RE.match(text_item.text):
                    continue
                if text_item.bbox.y0 >= item.bbox.y1 - 2:
                    distance = max(text_item.bbox.y0 - item.bbox.y1, 0)
                    if distance <= 140:
                        below.append((distance, text_item))
                elif text_item.bbox.y1 <= item.bbox.y0 + 2:
                    distance = max(item.bbox.y0 - text_item.bbox.y1, 0)
                    if distance <= 140:
                        above.append((distance, text_item))

            preferred = below if item.type == ContentType.IMAGE else above
            fallback = above if item.type == ContentType.IMAGE else below
            matches = preferred or fallback
            if matches:
                matches.sort(key=lambda pair: pair[0])
                item.captions = [matches[0][1].text]
