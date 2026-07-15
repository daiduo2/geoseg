from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "geoseg"


def _python_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _python_files_under(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")))
    return files


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_product_code_does_not_import_legacy_full_pipeline_directly():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines.full_pipeline"

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.name == "full_pipeline.py":
            continue

        if forbidden in _imported_modules(path):
            offenders.append(str(rel))

    assert offenders == []


def test_full_pipeline_remains_compatibility_only():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "full_pipeline.py"
    )

    assert "geoseg.modules.segment_engines.compat.full_pipeline" in imports
    assert not any(
        imported.startswith("geoseg.modules.segment_engines")
        and imported
        not in {
            "geoseg.modules.segment_engines.full_pipeline",
            "geoseg.modules.segment_engines.compat.full_pipeline",
        }
        for imported in imports
    )


def test_full_pipeline_process_figure_forwards_to_segmentation_stage(monkeypatch):
    from geoseg.modules.segment_engines.compat.full_pipeline import process_figure

    captured: dict[str, object] = {}

    def fake_run_segmentation_stage(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(
        "geoseg.modules.segment_engines.compat.full_pipeline.run_segmentation_stage",
        fake_run_segmentation_stage,
    )

    result = process_figure(
        np.zeros((8, 8, 3), dtype=np.uint8),
        caption="cap",
        text_blocks=[{"text": "cap"}],
        n_layers=4,
        quality_preference="best",
        skip_non_velocity_model=False,
        use_vlm=False,
        target_panel_id=3,
    )

    assert result == {"status": "ok"}
    assert captured["kwargs"] == {
        "caption": "cap",
        "text_blocks": [{"text": "cap"}],
        "n_layers": 4,
        "quality_preference": "best",
        "skip_non_velocity_model": False,
        "use_vlm": False,
        "target_panel_id": 3,
    }


def test_gui_main_window_uses_segment_engine_package_entrypoint():
    imports = _imported_modules(SRC_ROOT / "gui" / "main_window.py")

    assert "geoseg.modules.segment_engines" in imports
    assert "geoseg.modules.segment_engines.router" not in imports


def test_api_app_module_remains_app_assembly_only():
    imports = _imported_modules(SRC_ROOT / "api" / "app.py")

    assert "geoseg.api.routes_agent" in imports
    assert "geoseg.api.routes_manual" in imports
    assert "geoseg.api.routes_export" in imports
    assert "geoseg.api.routes_pdf" in imports
    assert not any(imported.startswith("geoseg.modules") for imported in imports)


def test_api_routes_use_segment_engine_package_entrypoint():
    imports = set()
    for path in [
        SRC_ROOT / "api" / "routes_agent.py",
        SRC_ROOT / "api" / "routes_manual.py",
    ]:
        imports.update(_imported_modules(path))

    assert "geoseg.modules.segment_engines" in imports
    assert "geoseg.modules.segment_engines.router" not in imports


def test_server_module_remains_thin_entrypoint():
    imports = _imported_modules(SRC_ROOT / "server.py")

    assert "geoseg.api.app" in imports
    assert not any(imported.startswith("geoseg.modules") for imported in imports)


def test_batch_processor_module_remains_thin_entrypoint():
    imports = _imported_modules(SRC_ROOT / "batch_processor.py")

    assert "geoseg.batch.cli" in imports
    assert "geoseg.batch.service" in imports
    assert not any(imported.startswith("geoseg.modules") for imported in imports)


def test_product_code_uses_segment_engine_package_entrypoint_not_router():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines.router"

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.name in {"__init__.py", "router.py"} and "segment_engines" in path.parts:
            continue

        if forbidden in _imported_modules(path):
            offenders.append(str(rel))

    assert offenders == []


def test_product_code_does_not_import_segment_engine_compat_directly():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines.compat"
    allowed_facades = {
        "_shared.py",
        "classify.py",
        "detect.py",
        "full_pipeline.py",
        "panel_segment.py",
        "pipeline_stages.py",
        "review.py",
        "summary.py",
    }

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.is_relative_to(SRC_ROOT / "modules" / "segment_engines" / "compat"):
            continue
        if (
            path.parent == SRC_ROOT / "modules" / "segment_engines"
            and path.name in allowed_facades
        ):
            continue

        if any(
            imported == forbidden or imported.startswith(f"{forbidden}.")
            for imported in _imported_modules(path)
        ):
            offenders.append(str(rel))

    assert offenders == []


def test_product_code_outside_segment_engines_does_not_import_segment_engine_internal():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines.internal"

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.is_relative_to(SRC_ROOT / "modules" / "segment_engines"):
            continue

        if any(
            imported == forbidden or imported.startswith(f"{forbidden}.")
            for imported in _imported_modules(path)
        ):
            offenders.append(str(rel))

    assert offenders == []


def test_visual_audit_report_uses_diagnostics_metrics_shim():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "visual_audit" / "report.py"
    )

    assert "geoseg.modules.segment_engines.metrics" in imports
    assert "geoseg.modules.segment_engines.diagnostics.metrics" not in imports
    assert "geoseg.modules.visual_audit.rendering" in imports
    assert "geoseg.modules.segment_engines.regional_fusion" not in imports


def test_preprocessing_pipeline_uses_local_segmentation_facade():
    imports = _imported_modules(SRC_ROOT / "preprocessing" / "pipeline.py")

    assert "geoseg.preprocessing.segmentation" in imports
    assert "geoseg.modules.segment_engines.v4_kmeans" not in imports
    assert "geoseg.modules.segment_engines.regional_fusion" not in imports


def test_preprocessing_segmentation_uses_public_engine_and_rendering_facades():
    imports = _imported_modules(SRC_ROOT / "preprocessing" / "segmentation.py")

    assert "geoseg.modules.segment_engines" in imports
    assert "geoseg.modules.visual_audit.rendering" in imports
    assert "geoseg.modules.segment_engines.v4_kmeans" not in imports
    assert "geoseg.modules.segment_engines.regional_fusion" not in imports


def test_visual_audit_views_use_local_rendering_facade():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "visual_audit" / "views.py"
    )

    assert "geoseg.modules.visual_audit.rendering" in imports
    assert "geoseg.modules.segment_engines.internal.regions" not in imports
    assert "geoseg.modules.segment_engines.v4_kmeans" not in imports


def test_visual_audit_rendering_wraps_segment_engine_rendering_helpers():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "visual_audit" / "rendering.py"
    )

    assert "geoseg.modules.segment_engines.regional_fusion" in imports
    assert "geoseg.modules.segment_engines.regions" in imports
    assert "geoseg.modules.segment_engines.internal.regions" not in imports


def test_regional_fusion_module_remains_compatibility_facade():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "regional_fusion.py"
    )

    assert "geoseg.modules.segment_engines.regional" in imports
    assert "geoseg.modules.post_process.split" not in imports
    assert "geoseg.modules.segment_engines.runner" not in imports


def test_segment_engine_region_helpers_are_exposed_through_public_facade():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "regions.py"
    )

    assert "geoseg.modules.segment_engines.internal.regions" in imports


def test_pipeline_stages_remains_compatibility_only():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines.pipeline_stages"

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.name == "pipeline_stages.py":
            continue

        if forbidden in _imported_modules(path):
            offenders.append(str(rel))

    assert offenders == []


def test_segmentation_orchestrator_uses_pipeline_stage_facade():
    imports = _imported_modules(SRC_ROOT / "pipeline" / "segment.py")

    assert "geoseg.pipeline.stages" in imports
    assert not any(
        imported.startswith("geoseg.modules.segment_engines")
        for imported in imports
    )


def test_pipeline_stage_helpers_do_not_reexport_legacy_stage_modules():
    forbidden = {
        "geoseg.modules.segment_engines.classify",
        "geoseg.modules.segment_engines.detect",
        "geoseg.modules.segment_engines.panel_segment",
        "geoseg.modules.segment_engines.review",
        "geoseg.modules.segment_engines.summary",
    }
    imports: set[str] = set()
    for path in sorted((SRC_ROOT / "pipeline" / "stages").rglob("*.py")):
        imports.update(_imported_modules(path))

    assert imports.isdisjoint(forbidden)


def test_runner_loads_engine_callables_from_registry():
    from geoseg.modules.segment_engines.registry import (
        ENGINE_REGISTRY,
        load_engine_callable,
    )

    for spec in ENGINE_REGISTRY.values():
        assert callable(load_engine_callable(spec)), spec.name


def test_runner_does_not_import_concrete_engine_modules_directly():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "runner.py"
    )
    forbidden = {
        "geoseg.modules.segment_engines.edge_grow",
        "geoseg.modules.segment_engines.edge_guided",
        "geoseg.modules.segment_engines.ensemble",
        "geoseg.modules.segment_engines.grayscale",
        "geoseg.modules.segment_engines.horizon_refinement",
        "geoseg.modules.segment_engines.kmeans_full",
        "geoseg.modules.segment_engines.tubular_structure",
        "geoseg.modules.segment_engines.v4_kmeans",
    }

    assert imports.isdisjoint(forbidden)


def test_edge_engines_use_shared_edge_helpers():
    edge_guided_imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "edge_guided.py"
    )
    edge_grow_imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "edge_grow.py"
    )
    forbidden = {
        "geoseg.modules.segment_engines.internal.seeds",
        "geoseg.modules.segment_engines.internal.regions",
    }

    assert any(
        imported.startswith("geoseg.modules.segment_engines.edge")
        for imported in edge_guided_imports
    )
    assert any(
        imported.startswith("geoseg.modules.segment_engines.edge")
        for imported in edge_grow_imports
    )
    assert edge_guided_imports.isdisjoint(forbidden)
    assert edge_grow_imports.isdisjoint(forbidden)


def test_strategy_memory_module_remains_compatibility_facade():
    imports = _imported_modules(
        SRC_ROOT / "modules" / "segment_engines" / "strategy_memory.py"
    )

    assert "geoseg.modules.segment_engines.strategy" in imports
    assert "geoseg.modules.segment_engines.strategy.records" in imports
    assert "json" not in imports
    assert "numpy" not in imports


def test_diagnostics_use_segmentation_stage_facade():
    offenders: list[str] = []
    forbidden_prefix = "geoseg.modules.segment_engines.full_pipeline"

    for path in sorted((SRC_ROOT / "modules" / "segment_engines" / "diagnostics").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        imports = _imported_modules(path)
        if forbidden_prefix in imports:
            offenders.append(str(rel))

    assert offenders == []


def test_product_code_imports_contracts_from_core_models():
    offenders: list[str] = []
    forbidden = "geoseg.pipeline_interfaces"

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.name == "pipeline_interfaces.py":
            continue

        if forbidden in _imported_modules(path):
            offenders.append(str(rel))

    assert offenders == []


def test_segment_engines_use_focused_internal_helpers():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines.internal.shared"

    for path in _python_files():
        rel = path.relative_to(REPO_ROOT)
        if path.name == "_shared.py":
            continue
        if path.name == "shared.py" and "compat" in path.parts:
            continue
        if path.name == "shared.py" and "internal" in path.parts:
            continue

        if forbidden in _imported_modules(path):
            offenders.append(str(rel))

    assert offenders == []


def test_scripts_and_examples_do_not_import_legacy_segment_engine_shared_shim():
    offenders: list[str] = []
    forbidden = "geoseg.modules.segment_engines._shared"

    for path in _python_files_under(REPO_ROOT / "scripts", REPO_ROOT / "examples"):
        rel = path.relative_to(REPO_ROOT)
        if forbidden in _imported_modules(path):
            offenders.append(str(rel))

    assert offenders == []


def test_parallel_segment_uses_public_runner_not_concrete_engines():
    imports = _imported_modules(REPO_ROOT / "scripts" / "parallel_segment.py")

    assert "geoseg.experiments" in imports
    assert "geoseg.modules.segment_engines.runner" not in imports
    assert "geoseg.modules.segment_engines.v4_kmeans" not in imports
    assert "geoseg.modules.segment_engines.ensemble" not in imports
    assert "geoseg.modules.segment_engines.grayscale" not in imports


def test_scripts_use_experiment_facade_for_common_cv_vlm_and_engine_helpers():
    offenders: list[str] = []
    forbidden = {
        "geoseg.modules.cv_detect.panel_detector",
        "geoseg.modules.vlm_client.client",
        "geoseg.modules.segment_engines.vlm_reps",
        "geoseg.modules.segment_engines.runner",
        "geoseg.modules.segment_engines.metrics",
        "geoseg.modules.segment_engines.strategy_memory",
    }

    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        imports = _imported_modules(path)
        if imports & forbidden:
            offenders.append(str(rel))

    assert offenders == []


def test_core_does_not_depend_on_feature_modules():
    offenders: list[str] = []
    forbidden_prefix = "geoseg.modules"

    for path in sorted((SRC_ROOT / "core").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        for imported in _imported_modules(path):
            if imported == forbidden_prefix or imported.startswith(f"{forbidden_prefix}."):
                offenders.append(str(rel))
                break

    assert offenders == []
