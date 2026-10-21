"""Persistence and label import for the curve editor.

A session is an ``.npz`` containing the curve model JSON, the explicit
background mask, and (when still untouched) the original labels.  JSON-only
export is provided for agents and review diffs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geoseg.modules.curve_editor.extract import curves_from_labels
from geoseg.modules.curve_editor.model import ControlPoint, InterfaceModel


def model_from_labels(
    labels: np.ndarray,
    *,
    background_mask: np.ndarray | None = None,
    extract: bool = True,
    tolerance: float = 1.5,
    max_points: int = 28,
) -> InterfaceModel:
    """Build an editable model from a label image.

    The extracted curves are marked ``initial``, so saving immediately returns
    the original labels exactly.  The first real edit switches to rasterizing
    the curve network.
    """
    labels = np.asarray(labels, dtype=np.int32)
    background = (
        np.asarray(background_mask, dtype=bool)
        if background_mask is not None
        else labels == 0
    )
    model = InterfaceModel(
        labels.shape,
        background_mask=background,
        base_labels=labels,
        merge_unseeded=True,
    )
    if not extract:
        return model
    for level, curve in enumerate(
        curves_from_labels(labels, tolerance=tolerance, max_points=max_points)
    ):
        model.add_interface(
            curve.points,
            level=level,
            closed=curve.closed,
            # B1 follows the extracted pixels closely; B3 is for smooth
            # manual editing after import.
            basis="B1",
            initial=True,
        )
    return model


def model_to_dict(model: InterfaceModel) -> dict:
    return model.to_dict()


def model_from_dict(
    data: dict,
    *,
    background_mask: np.ndarray | None = None,
    base_labels: np.ndarray | None = None,
) -> InterfaceModel:
    model = InterfaceModel(
        tuple(data["shape"]),
        background_mask=background_mask,
        base_labels=base_labels,
        merge_unseeded=bool(data.get("merge_unseeded", False)),
    )
    for interface_data in data.get("interfaces", []):
        points = [
            ControlPoint(
                id=point["id"],
                x=point["x"],
                y=point["y"],
                kind=point.get("kind", "free"),
                locked=point.get("locked", False),
            )
            for point in interface_data["points"]
        ]
        interface = model.add_interface(
            points,
            level=interface_data.get("level", 0),
            closed=interface_data.get("closed", False),
            basis=interface_data.get("basis", "B3"),
            point_mode=interface_data.get("point_mode", "knots"),
            initial=True,
            interface_id=interface_data["id"],
        )
        interface.fit_status = interface_data.get("fit_status", "ok")
        interface.confidence = float(interface_data.get("confidence", 1.0))
        interface.metadata = dict(interface_data.get("metadata", {}))
    if data.get("regions"):
        model.regions = {
            int(region_id): (float(seed[0]), float(seed[1]))
            for region_id, seed in data["regions"].items()
        }
        model._next_region = max(model.regions, default=0) + 1
    return model


def save_session(model: InterfaceModel, path: str | Path) -> Path:
    """Save curves, background mask, and untouched base labels to an npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_json": np.asarray(json.dumps(model.to_dict(), ensure_ascii=False)),
        "background_mask": np.asarray(model.background_mask, dtype=bool),
    }
    if model.base_labels is not None:
        payload["base_labels"] = np.asarray(model.base_labels, dtype=np.int32)
    np.savez_compressed(path, **payload)
    return path


def load_session(path: str | Path) -> InterfaceModel:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        model_json = str(data["model_json"])
        background = np.asarray(data["background_mask"], dtype=bool)
        base = np.asarray(data["base_labels"], dtype=np.int32) if "base_labels" in data else None
    return model_from_dict(json.loads(model_json), background_mask=background, base_labels=base)


def save_shapes(model: InterfaceModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_shapes(path: str | Path) -> InterfaceModel:
    path = Path(path)
    return model_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_labels(model: InterfaceModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=model.to_labels())
    return path


def load_labels(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        key = "labels" if "labels" in data else list(data.keys())[0]
        return np.asarray(data[key])


__all__ = [
    "load_labels",
    "load_session",
    "load_shapes",
    "model_from_dict",
    "model_from_labels",
    "model_to_dict",
    "save_labels",
    "save_session",
    "save_shapes",
]
