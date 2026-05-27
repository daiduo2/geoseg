"""VLM client: prompt assembly + Claude CLI call + JSON validation + audit.

This is the ONLY LLM exit point in geoseg. Calls the local `claude` CLI
in non-interactive mode (`-p`) with `--json-schema` for structured output.
No external API key is managed here — the CLI uses the user's existing
Claude Code session/auth.

Test scenario:
    >>> from geoseg.modules.vlm_client.client import review_page_overview
    >>> import numpy as np
    >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    >>> result = review_page_overview(img, [], page_idx=7, mode="stub")
    >>> assert isinstance(result, dict)  # stub returns prompt dict
    >>> # mode="auto" would call the Claude CLI (costs API tokens)
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from geoseg.pipeline_interfaces import QualityReview

from .prompts import (
    VERSION,
    PAGE_OVERVIEW_PROMPT,
    FIGURE_CLASSIFICATION_PROMPT,
    SEGMENTATION_QUALITY_PROMPT,
    PageOverview,
    FigureClassification,
    SegmentationQualityReview,
)


DEFAULT_AUDIT_DIR = Path(__file__).resolve().parents[3] / "runs" / "audit"
MIN_CONFIDENCE = 0.7
MAX_RETRY = 3
PAGE_BUDGET = 5


# ── Audit ──────────────────────────────────────────────────────────

def _write_audit(
    step: str,
    prompt: str,
    model_version: str,
    input_images: list[Path],
    output_json: dict,
    confidence: float,
    audit_dir: Path,
    retry_count: int = 0,
    api_metadata: dict | None = None,
) -> Path:
    """Write audit JSON to disk. Returns the audit file path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{ts}_{step}"
    if retry_count > 0:
        fname += f"_retry{retry_count}"
    fname += ".json"
    audit_path = audit_dir / fname

    audit = {
        "timestamp": ts,
        "step": step,
        "prompt_version": VERSION,
        "model_version": model_version,
        "input_images": [str(p) for p in input_images],
        "output_json": output_json,
        "confidence": confidence,
        "retry_count": retry_count,
    }
    if api_metadata:
        audit["api_metadata"] = api_metadata

    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit_path


def _save_image_for_audit(img: np.ndarray, audit_dir: Path, prefix: str) -> Path:
    """Save numpy array as PNG for audit trail. Returns absolute path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = audit_dir / f"{prefix}_{ts}.png"
    Image.fromarray(img).save(path)
    return path.resolve()


# ── Claude CLI wrapper ─────────────────────────────────────────────

def _call_claude_cli(
    prompt: str,
    json_schema: dict,
    timeout: int = 300,
) -> tuple[dict, dict]:
    """DEPRECATED: Use `.claude/skills/figure-classify` or `.claude/skills/sandbox-segment`
    instead. This function calls `claude -p` via subprocess, which is the old
    Python-client pattern. The project has moved to agent-native skills.

    Call `claude -p` with structured JSON output.

    Args:
        prompt: The full prompt text. Any absolute file paths in the prompt
                are automatically read by the Claude CLI as image resources.
        json_schema: JSON Schema dict for structured output validation.
        timeout: Subprocess timeout in seconds.

    Returns:
        (structured_output, full_wrapper) where full_wrapper contains
        api_metadata like cost_usd, token usage, session_id.

    Raises:
        RuntimeError: if claude CLI is not found, exits non-zero, or returns
                      no structured_output.
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError("claude CLI not found in PATH")

    cmd = [
        claude_path,
        "-p", prompt,
        "--json-schema", json.dumps(json_schema),
        "--output-format", "json",
        "--dangerously-skip-permissions",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).resolve().parents[3]),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={result.returncode}): {result.stderr[:500]}"
        )

    try:
        wrapper = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude CLI returned invalid JSON: {e}\nstdout={result.stdout[:500]}"
        ) from e

    if "structured_output" not in wrapper:
        raise RuntimeError(
            f"claude CLI returned no structured_output: {wrapper.get('result', 'unknown')}"
        )

    return wrapper["structured_output"], wrapper


def _call_with_retry(
    prompt: str,
    json_schema: dict,
    timeout: int = 300,
    max_retry: int = MAX_RETRY,
) -> tuple[dict, dict]:
    """DEPRECATED: Use agent skills instead of CLI subprocess calls.

    Call Claude CLI with retry budget.

    Retries on RuntimeError (CLI failure / bad JSON / missing structured_output).
    Does NOT retry on schema validation or confidence errors.
    """
    last_err: Exception | None = None
    for attempt in range(max_retry + 1):
        try:
            return _call_claude_cli(prompt, json_schema, timeout)
        except RuntimeError as exc:
            last_err = exc
            if attempt < max_retry:
                continue
    raise last_err or RuntimeError("Claude CLI call failed after all retries")


def _build_json_schema(schema_class: type) -> dict:
    """Build JSON Schema from pydantic model for Claude CLI --json-schema."""
    raw = schema_class.model_json_schema()
    # Ensure all required fields are present in schema
    if "required" not in raw:
        raw["required"] = list(raw.get("properties", {}).keys())
    return raw


# ── Public API ─────────────────────────────────────────────────────

def review_page_overview(
    image: np.ndarray,
    text_blocks: list[dict],
    page_idx: int = 0,
    audit_dir: Path | None = None,
    mode: Literal["auto", "stub"] = "auto",
    min_confidence: float | None = None,
) -> PageOverview | dict:
    """Review a page image for figure type, panels, and target selection (VLM-0).

    Args:
        image: RGB image array.
        text_blocks: List of {"bbox": [...], "text": "..."} from pdf_extractor.
        page_idx: Page number for context.
        audit_dir: Where to write audit JSON. Default: runs/audit/.
        mode: "auto" calls Claude CLI (costs tokens). "stub" returns prompt dict only.
        min_confidence: Override confidence threshold. Default: MIN_CONFIDENCE (0.7).

    Returns:
        PageOverview pydantic model (auto mode) or dict with prompt/audit_path (stub).

    Raises:
        RuntimeError: CLI failure.
        ValueError: Schema validation or confidence check failure.
    """
    if audit_dir is None:
        audit_dir = DEFAULT_AUDIT_DIR

    img_path = _save_image_for_audit(image, audit_dir, f"page_{page_idx}_overview_input")

    context = f"""Page index: {page_idx}
Image shape: {image.shape[0]}x{image.shape[1]} (HxW)
Text blocks on page ({len(text_blocks)}):
"""
    for tb in text_blocks[:15]:
        text = tb.get("text", "")
        bbox = tb.get("bbox", [])
        bbox_str = f" [bbox: {','.join(map(str, bbox))}]" if bbox else ""
        item_type = tb.get("type", "text")
        if text:
            context += f'  - [{item_type}]{bbox_str} "{text[:120]}"\n'

    prompt = PAGE_OVERVIEW_PROMPT + "\n\n" + context
    prompt += f"\n\nAnalyze the following image and respond with JSON:\nImage: {img_path}"

    if mode == "stub":
        audit_path = _write_audit(
            step="page_overview",
            prompt=prompt,
            model_version="claude-sonnet-4-6",
            input_images=[img_path],
            output_json={},
            confidence=0.0,
            audit_dir=audit_dir,
        )
        return {
            "prompt": prompt,
            "audit_path": audit_path,
            "schema_class": PageOverview,
            "image_audit_path": img_path,
            "step": "page_overview",
        }

    # Auto mode: call Claude CLI
    schema = _build_json_schema(PageOverview)
    structured, wrapper = _call_with_retry(prompt, schema)

    instance = PageOverview.model_validate(structured)

    # Audit FIRST — even if confidence is low (for post-mortem analysis)
    api_meta = {
        "cost_usd": wrapper.get("total_cost_usd"),
        "input_tokens": wrapper.get("usage", {}).get("input_tokens"),
        "output_tokens": wrapper.get("usage", {}).get("output_tokens"),
        "session_id": wrapper.get("session_id"),
        "duration_ms": wrapper.get("duration_ms"),
    }
    _write_audit(
        step="page_overview",
        prompt=prompt,
        model_version=wrapper.get("modelUsage", {}).get("claude-sonnet-4-6", {}).get("model", "claude-sonnet-4-6"),
        input_images=[img_path],
        output_json=structured,
        confidence=instance.confidence,
        audit_dir=audit_dir,
        api_metadata=api_meta,
    )

    _threshold = min_confidence if min_confidence is not None else MIN_CONFIDENCE
    if instance.confidence < _threshold:
        raise ValueError(
            f"Confidence too low: {instance.confidence} < {_threshold} (page_overview)"
        )

    return instance


def classify_figure(
    image: np.ndarray,
    audit_dir: Path | None = None,
    mode: Literal["auto", "stub"] = "auto",
    min_confidence: float | None = None,
) -> FigureClassification | dict:
    """Semantic figure classification using VLM (VLM-C).

    This is the SECOND-LINE filter after the fast CV heuristic classifier.
    Only "velocity_model" and "geological_cross_section" should proceed to segmentation.

    Args:
        image: RGB image array of the extracted figure.
        audit_dir: Where to write audit JSON. Default: runs/audit/.
        mode: "auto" calls Claude CLI (costs tokens). "stub" returns prompt dict only.
        min_confidence: Override confidence threshold. Default: MIN_CONFIDENCE (0.7).

    Returns:
        FigureClassification model (auto mode) or dict with prompt/audit_path (stub).

    Raises:
        RuntimeError: CLI failure.
        ValueError: Schema validation or confidence check failure.
    """
    if audit_dir is None:
        audit_dir = DEFAULT_AUDIT_DIR

    img_path = _save_image_for_audit(image, audit_dir, "figure_classification_input")

    context = f"""Image shape: {image.shape[0]}x{image.shape[1]} (HxW)
"""

    prompt = FIGURE_CLASSIFICATION_PROMPT + "\n\n" + context
    prompt += f"\n\nClassify the following image and respond with JSON:\nImage: {img_path}"

    if mode == "stub":
        audit_path = _write_audit(
            step="figure_classification",
            prompt=prompt,
            model_version="claude-sonnet-4-6",
            input_images=[img_path],
            output_json={},
            confidence=0.0,
            audit_dir=audit_dir,
        )
        return {
            "prompt": prompt,
            "audit_path": audit_path,
            "schema_class": FigureClassification,
            "image_audit_path": img_path,
            "step": "figure_classification",
        }

    schema = _build_json_schema(FigureClassification)
    structured, wrapper = _call_with_retry(prompt, schema)

    instance = FigureClassification.model_validate(structured)

    api_meta = {
        "cost_usd": wrapper.get("total_cost_usd"),
        "input_tokens": wrapper.get("usage", {}).get("input_tokens"),
        "output_tokens": wrapper.get("usage", {}).get("output_tokens"),
        "session_id": wrapper.get("session_id"),
        "duration_ms": wrapper.get("duration_ms"),
    }
    _write_audit(
        step="figure_classification",
        prompt=prompt,
        model_version=wrapper.get("modelUsage", {}).get("claude-sonnet-4-6", {}).get("model", "claude-sonnet-4-6"),
        input_images=[img_path],
        output_json=structured,
        confidence=instance.confidence,
        audit_dir=audit_dir,
        api_metadata=api_meta,
    )

    _threshold = min_confidence if min_confidence is not None else MIN_CONFIDENCE
    if instance.confidence < _threshold:
        raise ValueError(
            f"Confidence too low: {instance.confidence} < {_threshold} (figure_classification)"
        )

    return instance


def review_segmentation_quality(
    image: np.ndarray,
    audit_dir: Path | None = None,
    mode: Literal["auto", "stub"] = "auto",
    min_confidence: float | None = None,
) -> SegmentationQualityReview | dict:
    """Review segmentation quality by comparing original + overlay (VLM-SQ).

    Args:
        image: Side-by-side composite image (left=original, right=segmentation).
        audit_dir: Where to write audit JSON. Default: runs/audit/.
        mode: "auto" calls Claude CLI. "stub" returns prompt dict only.
        min_confidence: Override confidence threshold. Default: MIN_CONFIDENCE (0.7).

    Returns:
        SegmentationQualityReview model (auto mode) or dict with prompt/audit_path (stub).
    """
    if audit_dir is None:
        audit_dir = DEFAULT_AUDIT_DIR

    img_path = _save_image_for_audit(image, audit_dir, "segmentation_quality_input")

    prompt = SEGMENTATION_QUALITY_PROMPT + f"\n\nReview the following image:\nImage: {img_path}"

    if mode == "stub":
        audit_path = _write_audit(
            step="segmentation_quality",
            prompt=prompt,
            model_version="claude-sonnet-4-6",
            input_images=[img_path],
            output_json={},
            confidence=0.0,
            audit_dir=audit_dir,
        )
        return {
            "prompt": prompt,
            "audit_path": audit_path,
            "schema_class": SegmentationQualityReview,
            "image_audit_path": img_path,
            "step": "segmentation_quality",
        }

    schema = _build_json_schema(SegmentationQualityReview)
    structured, wrapper = _call_with_retry(prompt, schema)

    instance = SegmentationQualityReview.model_validate(structured)

    api_meta = {
        "cost_usd": wrapper.get("total_cost_usd"),
        "input_tokens": wrapper.get("usage", {}).get("input_tokens"),
        "output_tokens": wrapper.get("usage", {}).get("output_tokens"),
        "session_id": wrapper.get("session_id"),
        "duration_ms": wrapper.get("duration_ms"),
    }
    _write_audit(
        step="segmentation_quality",
        prompt=prompt,
        model_version=wrapper.get("modelUsage", {}).get("claude-sonnet-4-6", {}).get("model", "claude-sonnet-4-6"),
        input_images=[img_path],
        output_json=structured,
        confidence=instance.overall_score,
        audit_dir=audit_dir,
        api_metadata=api_meta,
    )

    _threshold = min_confidence if min_confidence is not None else MIN_CONFIDENCE
    if instance.overall_score < _threshold:
        raise ValueError(
            f"Segmentation quality too low: {instance.overall_score} < {_threshold}"
        )

    return instance


# ── QualityReviewer Protocol adapter ───────────────────────────────

def quality_review(
    img_rgb: np.ndarray,
    context: dict | None = None,
) -> QualityReview:
    """QualityReviewer Protocol adapter for vlm_client.

    Routes to ``classify_figure`` or ``review_page_overview`` based on
    ``context["review_type"]`` and returns a standardized QualityReview dict.

    Args:
        img_rgb: Figure image.
        context: Optional dict with keys:
            - ``review_type``: ``"figure_classification"`` (default) or ``"page_overview"``
            - ``text_blocks``: list of text blocks (for page_overview)
            - ``page_idx``: page index (default 0)
            - ``mode``: ``"auto"`` or ``"stub"`` (default ``"auto"``)
            - ``min_confidence``: override threshold

    Returns:
        QualityReview dict.
    """
    ctx = context or {}
    review_type = ctx.get("review_type", "figure_classification")
    mode = ctx.get("mode", "auto")
    min_conf = ctx.get("min_confidence")

    if review_type == "figure_classification":
        result = classify_figure(img_rgb, mode=mode, min_confidence=min_conf)
        if isinstance(result, dict) and "prompt" in result:
            # stub mode returns prompt dict, not pydantic model
            return {
                "warnings": ["stub_mode"],
                "score": 0.0,
                "can_auto_fix": False,
                "suggested_action": "continue",
            }
        return {
            "warnings": [],
            "score": result.confidence,
            "can_auto_fix": False,
            "suggested_action": (
                "continue"
                if getattr(result, "segmentation_recommendation", None) == "proceed"
                else "skip"
            ),
        }

    if review_type == "page_overview":
        result = review_page_overview(
            img_rgb,
            text_blocks=ctx.get("text_blocks", []),
            page_idx=ctx.get("page_idx", 0),
            mode=mode,
            min_confidence=min_conf,
        )
        if isinstance(result, dict) and "prompt" in result:
            # stub mode returns prompt dict, not pydantic model
            return {
                "warnings": ["stub_mode"],
                "score": 0.0,
                "can_auto_fix": False,
                "suggested_action": "continue",
            }
        warnings: list[str] = []
        if hasattr(result, "panels") and len(result.panels) == 0:
            warnings.append("no_panels_detected")
        return {
            "warnings": warnings,
            "score": result.confidence,
            "can_auto_fix": len(warnings) > 0,
            "suggested_action": (
                "continue" if result.confidence >= MIN_CONFIDENCE else "manual_intervention"
            ),
        }

    raise ValueError(f"Unknown review_type: {review_type}")


# ── Legacy parse helper (kept for backward compat / manual mode) ───

def parse_response(raw_json: dict, schema_class: type, min_confidence: float = MIN_CONFIDENCE) -> Any:
    """Parse and validate a VLM JSON response against a pydantic schema.

    Args:
        raw_json: The JSON dict returned by the VLM.
        schema_class: One of PageOverview, FigureClassification.
        min_confidence: Minimum confidence threshold (default 0.7).

    Returns:
        Validated pydantic model instance.

    Raises:
        ValueError: If schema validation fails or confidence < min_confidence.
    """
    try:
        instance = schema_class.model_validate(raw_json)
    except Exception as e:
        raise ValueError(f"Schema validation failed for {schema_class.__name__}: {e}") from e

    if instance.confidence < min_confidence:
        raise ValueError(
            f"Confidence too low: {instance.confidence} < {min_confidence} "
            f"({schema_class.__name__})"
        )

    return instance
