"""Background worker for PDF import via MinerU API.

Runs in a QThread so the GUI stays responsive during upload/polling/download.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from urllib.request import urlopen

from PySide6.QtCore import QThread, Signal


def _parse_mineru_page_map(extracted_dir: Path) -> dict[str, int]:
    """Parse MinerU content_list_v2.json to build figure -> page index mapping.

    MinerU v4 stores figure metadata in a JSON file alongside the images.
    Image filenames are hashes, so we must read the JSON to know which page
    each figure came from.
    """
    page_map: dict[str, int] = {}

    # Find the content_list_v2.json file
    candidates = list(extracted_dir.glob("*_content_list_v2.json"))
    if not candidates:
        candidates = list(extracted_dir.glob("*_content_list.json"))
    if not candidates:
        return page_map

    json_path = candidates[0]
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return page_map

    if not isinstance(data, list):
        return page_map

    images_dir = extracted_dir / "images"

    for page_idx, page_items in enumerate(data):
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            content = item.get("content", {})
            if not isinstance(content, dict):
                continue
            img_source = content.get("image_source", {})
            rel_path = img_source.get("path", "")
            if not rel_path:
                continue
            # Fix truncated extensions that sometimes appear in JSON
            rel_path = rel_path.rstrip(".")
            if not rel_path.endswith(".jpg") and not rel_path.endswith(".png"):
                # Try to match against actual files in images dir
                base_name = Path(rel_path).name
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = images_dir / (base_name + ext if "." not in base_name else base_name)
                    if candidate.exists():
                        page_map[str(candidate)] = page_idx
                        break
            else:
                full_path = images_dir / Path(rel_path).name
                if full_path.exists():
                    page_map[str(full_path)] = page_idx

    return page_map


def _parse_mineru_caption_map(extracted_dir: Path) -> dict[str, str]:
    """Parse MinerU content_list_v2.json to build figure -> caption text mapping.

    Extracts image captions from the JSON metadata so VLM review_page_overview
    receives actual figure captions rather than empty text_blocks.
    """
    caption_map: dict[str, str] = {}

    candidates = list(extracted_dir.glob("*_content_list_v2.json"))
    if not candidates:
        candidates = list(extracted_dir.glob("*_content_list.json"))
    if not candidates:
        return caption_map

    json_path = candidates[0]
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return caption_map

    if not isinstance(data, list):
        return caption_map

    images_dir = extracted_dir / "images"

    def _resolve_image_path(rel_path: str) -> Path | None:
        """Resolve a relative image path to an absolute path in images_dir."""
        if not rel_path:
            return None
        rel_path = rel_path.rstrip(".")
        if rel_path.endswith(".jpg") or rel_path.endswith(".png"):
            p = images_dir / Path(rel_path).name
            return p if p.exists() else None
        base_name = Path(rel_path).name
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (base_name + ext if "." not in base_name else base_name)
            if candidate.exists():
                return candidate
        return None

    def _extract_caption_text(caption_items: list) -> str:
        """Extract plain text from MinerU caption structure."""
        parts: list[str] = []
        for ci in caption_items:
            if isinstance(ci, dict) and ci.get("type") == "text":
                text = ci.get("content", "")
                if text:
                    parts.append(text)
        return " ".join(parts).strip()

    for page_items in data:
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            content = item.get("content", {})
            if not isinstance(content, dict):
                continue
            img_source = content.get("image_source", {})
            rel_path = img_source.get("path", "")
            img_path = _resolve_image_path(rel_path)
            if img_path is None:
                continue
            caption_items = content.get("image_caption", [])
            caption_text = _extract_caption_text(caption_items)
            if caption_text:
                caption_map[str(img_path)] = caption_text

    return caption_map


def _parse_mineru_text_blocks(extracted_dir: Path) -> dict[int, list[dict]]:
    """Parse MinerU content_list to build page_idx -> text_blocks mapping.

    Each text block contains spatial context:
        {"type": str, "text": str, "bbox": [x1, y1, x2, y2]}

    Includes titles, paragraphs, and image captions/footnotes with bboxes.
    Position info helps VLM understand figure-context relationships.
    """
    text_blocks_map: dict[int, list[dict]] = {}

    candidates = list(extracted_dir.glob("*_content_list_v2.json"))
    if not candidates:
        candidates = list(extracted_dir.glob("*_content_list.json"))
    if not candidates:
        return text_blocks_map

    json_path = candidates[0]
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return text_blocks_map

    if not isinstance(data, list):
        return text_blocks_map

    def _extract_text_from_content(content: dict, content_key: str) -> str:
        parts: list[str] = []
        for item in content.get(content_key, []):
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("content", "")
                if t:
                    parts.append(t)
        return " ".join(parts)

    for page_idx, page_items in enumerate(data):
        blocks: list[dict] = []
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            bbox = item.get("bbox", [])

            if item_type == "image":
                content = item.get("content", {})
                caption_text = _extract_text_from_content(content, "image_caption")
                if caption_text:
                    blocks.append({"type": "image_caption", "text": caption_text, "bbox": bbox})
                footnote_text = _extract_text_from_content(content, "image_footnote")
                if footnote_text:
                    blocks.append({"type": "image_footnote", "text": footnote_text, "bbox": bbox})
            elif item_type in ("title", "paragraph"):
                content = item.get("content", {})
                content_key = f"{item_type}_content"
                text = _extract_text_from_content(content, content_key)
                if text:
                    blocks.append({"type": item_type, "text": text, "bbox": bbox})
            elif item_type == "text":
                text = item.get("content", "")
                if isinstance(text, str) and text:
                    blocks.append({"type": "text", "text": text, "bbox": bbox})

        if blocks:
            text_blocks_map[page_idx] = blocks

    return text_blocks_map


class PdfImportWorker(QThread):
    """Upload a PDF to MinerU, poll for results, download and unpack the zip."""

    progress = Signal(str)
    # (extracted_dir, [image_paths...], {image_path: page_idx}, {image_path: caption}, {page_idx: [text_blocks]})
    finished_success = Signal(str, list, dict, dict, dict)
    finished_error = Signal(str)

    def __init__(
        self,
        pdf_path: str,
        out_dir: str | None = None,
        poll_interval: int = 5,
        max_wait: int = 300,
    ):
        super().__init__()
        self._pdf_path = Path(pdf_path)
        self._out_dir = out_dir
        self._poll_interval = poll_interval
        self._max_wait = max_wait

    def run(self) -> None:
        try:
            self._do_import()
        except Exception as exc:
            self.finished_error.emit(str(exc))

    def _do_import(self) -> None:
        # Delayed import so the GUI can start without requests installed.
        from geoseg.modules.mineru_client import upload_and_extract

        self.progress.emit(f"Uploading {self._pdf_path.name} to MinerU ...")

        result = upload_and_extract(
            self._pdf_path,
            poll_interval=self._poll_interval,
            max_wait=self._max_wait,
        )

        zip_url = result.get("full_zip_url") or result.get("zip_url")
        if not zip_url:
            raise RuntimeError(
                f"MinerU result missing zip URL. Available keys: {list(result.keys())}"
            )

        self.progress.emit("Downloading extracted archive ...")
        with urlopen(zip_url, timeout=60) as resp:
            zip_data = resp.read()

        # Determine output directory
        if self._out_dir:
            out_path = Path(self._out_dir)
        else:
            out_path = Path(tempfile.gettempdir()) / f"geoseg_{self._pdf_path.stem}"
        out_path.mkdir(parents=True, exist_ok=True)

        zip_path = out_path / "extracted.zip"
        zip_path.write_bytes(zip_data)

        self.progress.emit("Unpacking archive ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)

        # Clean up zip to save space
        zip_path.unlink(missing_ok=True)

        # Find all images in the extracted directory
        images_dir = out_path / "images"
        if not images_dir.exists():
            candidates = list(out_path.rglob("images"))
            if candidates:
                images_dir = candidates[0]

        image_paths: list[str] = []
        if images_dir.exists():
            image_paths = sorted(
                str(p) for p in images_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )

        # Build page map, caption map, and text blocks from MinerU JSON metadata
        page_map = _parse_mineru_page_map(out_path)
        caption_map = _parse_mineru_caption_map(out_path)
        text_blocks_map = _parse_mineru_text_blocks(out_path)

        self.progress.emit(
            f"Extraction complete: {len(image_paths)} figures found, "
            f"{len(page_map)} mapped to pages, {len(caption_map)} with captions, "
            f"{len(text_blocks_map)} pages with text blocks."
        )
        self.finished_success.emit(str(out_path), image_paths, page_map, caption_map, text_blocks_map)
