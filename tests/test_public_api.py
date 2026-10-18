"""Public API and compatibility-boundary guardrails.

These tests enforce the navigation contract:

- ``geoseg.api`` is the only entry scripts should need.
- Product code outside ``geoseg.compat`` must not import the legacy layer.
- The full pipeline lives in ``geoseg.pipeline.run``; the compat controller
  is a forwarding shim, not a second implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from geoseg import api

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "geoseg"

REQUIRED_API = {
    "run_pipeline",
    "run_segmentation_stage",
    "run_post_process_and_export",
    "classify_figure",
    "detect_panels",
    "extract_colorbar",
    "run_engine",
    "route_and_segment",
    "segment_v4_kmeans",
    "compute_all",
    "create_overlay",
    "row_median_filter",
    "labels_to_shapes",
    "shapes_to_labels",
    "bridge_boundary_with_endpoint_tangents",
    "labels_to_grids",
    "write_tomography_file",
    "SegmentationResult",
    "PanelInput",
}

# Paper-specific drivers intentionally use algorithm internals; they are not
# part of the general script surface.
SCRIPT_DEEP_IMPORT_ALLOWLIST = {
    "process_j_tecto_2019_06_024.py",
    "process_j_tecto_2019_06_024_direct.py",
}

SCRIPT_ALLOWED_PREFIXES = (
    "geoseg.api",
    "geoseg.core",
    "geoseg.batch",
    "geoseg.session_state",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_api_exposes_required_names() -> None:
    exported = set(api.__all__)
    assert REQUIRED_API <= exported, f"missing from geoseg.api: {REQUIRED_API - exported}"
    for name in REQUIRED_API:
        assert getattr(api, name, None) is not None, name


def test_scripts_import_only_the_public_api() -> None:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        if path.name in SCRIPT_DEEP_IMPORT_ALLOWLIST:
            continue
        for module in _imported_modules(path):
            if module == "geoseg" or module.startswith("geoseg."):
                if not module.startswith(SCRIPT_ALLOWED_PREFIXES):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} -> {module}")
    assert offenders == []


def test_product_code_does_not_reach_into_compat() -> None:
    offenders: list[str] = []
    compat_root = SRC_ROOT / "compat"
    for path in SRC_ROOT.rglob("*.py"):
        if path.is_relative_to(compat_root) or "__pycache__" in path.parts:
            continue
        for module in _imported_modules(path):
            if module.startswith("geoseg.compat"):
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {module}")
    assert offenders == []


def test_experiments_facade_is_gone() -> None:
    assert not (SRC_ROOT / "experiments.py").exists()
    offenders: list[str] = []
    for root in ("src", "tests", "scripts", "examples", "experiments"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "experiments" in _imported_modules(path):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_compat_controller_forwards_to_pipeline_run() -> None:
    from geoseg.compat.controller import run_pipeline as compat_run
    from geoseg.pipeline.run import run_pipeline as canonical_run

    assert compat_run is canonical_run


@pytest.mark.parametrize("name", ["run_pipeline", "run_segmentation_stage"])
def test_pipeline_package_exports_orchestration(name: str) -> None:
    import geoseg.pipeline as pipeline

    assert name in pipeline.__all__
    assert callable(getattr(pipeline, name))


SINGLE_SOURCE_DEFINITIONS = {
    # canonical contracts
    "PanelInput": "core/models.py",
    "SegmentationMeta": "core/models.py",
    "SegmentationResult": "core/models.py",
    "RegionalAudit": "core/models.py",
    "FigureClassification": "core/models.py",
    "PageOverview": "core/models.py",
    # canonical image helpers
    "adaptive_blur": "core/image_ops.py",
    "row_median_filter": "core/image_ops.py",
    "estimate_noise_level": "core/image_ops.py",
    # canonical post-process / detection entry points
    "extract_components": "modules/post_process/polygon.py",
    "detect_panels": "modules/cv_detect/panel_detector.py",
}


def test_contracts_and_shared_helpers_have_one_definition() -> None:
    definitions: dict[str, list[str]] = {}
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                definitions.setdefault(node.name, []).append(
                    str(path.relative_to(SRC_ROOT))
                )

    offenders = {
        name: definitions.get(name, [])
        for name, expected in SINGLE_SOURCE_DEFINITIONS.items()
        if definitions.get(name, []) != [expected]
    }
    assert offenders == {}, f"duplicate definitions: {offenders}"
