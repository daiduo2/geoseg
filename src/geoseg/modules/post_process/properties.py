"""Map layer color names to elastic properties (Vp, Vs, rho).

The default table is a coarse crustal-scale template based on typical
continental velocity structures. Users override per-study via JSON.

Test scenario:
    >>> props = assign_properties(["red", "blue"])
    >>> assert "red" in props
    >>> assert props["red"]["Vp"] > props["blue"]["Vp"]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage


# Crustal-scale template — continental, no units enforced here
# (typical units: Vp/Vs in m/s or km/s; rho in kg/m3 or g/cm3)
DEFAULT_PROPERTIES = {
    "red": {"Vp": 6500.0, "Vs": 3750.0, "rho": 2800.0},
    "orange": {"Vp": 5500.0, "Vs": 3200.0, "rho": 2700.0},
    "yellow": {"Vp": 4500.0, "Vs": 2600.0, "rho": 2500.0},
    "green": {"Vp": 3500.0, "Vs": 2000.0, "rho": 2300.0},
    "blue": {"Vp": 2500.0, "Vs": 1500.0, "rho": 2100.0},
    "purple": {"Vp": 7500.0, "Vs": 4300.0, "rho": 3000.0},
}


def assign_properties(
    color_names: list[str],
    custom_map: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Return a {color_name: {"Vp": float, "Vs": float, "rho": float}} dict.

    Missing colors fall back to DEFAULT_PROPERTIES; if still missing,
    a ValueError is raised so the user must supply the mapping.
    """
    out = {}
    src = custom_map or {}
    for name in color_names:
        if name in src:
            out[name] = dict(src[name])
        elif name in DEFAULT_PROPERTIES:
            out[name] = dict(DEFAULT_PROPERTIES[name])
        else:
            raise ValueError(
                f"No property mapping for color '{name}'. "
                "Provide it via custom_map or a JSON file."
            )
    return out


def load_properties_json(path: str | Path) -> dict[str, dict]:
    """Load a user-supplied property table.

    Expected schema:
        {"red": {"Vp": 6500, "Vs": 3750, "rho": 2800}, ...}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for color, vals in data.items():
        for key in ("Vp", "Vs", "rho"):
            if key not in vals:
                raise ValueError(f"Property table entry '{color}' missing '{key}'")
    return data


def save_properties_json(props: dict[str, dict], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(props, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_properties_template(color_names: list[str]) -> dict[str, dict]:
    """Emit a template dict the user can edit and pass back."""
    return assign_properties(color_names)


def generate_properties_for_layers(color_names: list[str]) -> dict[str, dict]:
    """Auto-generate a property table for unknown color/layer names.

    Vp ranges from 3000 to 6000 m/s across layers, Vs ≈ Vp/√3,
    rho ranges from 2200 to 2800 kg/m³.
    """
    n = len(color_names)
    out: dict[str, dict] = {}
    for i, name in enumerate(color_names):
        t = i / max(1, n - 1) if n > 1 else 0.5
        vp = 3000.0 + t * 3000.0
        vs = vp / 1.732
        rho = 2200.0 + t * 600.0
        out[name] = {"Vp": round(vp, 2), "Vs": round(vs, 2), "rho": round(rho, 2)}
    return out


def inherit_properties_for_new_labels(
    labels: np.ndarray,
    base_properties: dict[str, dict],
    color_names: list[str],
) -> dict[str, dict]:
    """Extend property table when napari editing creates new labels.

    New labels (those not covered by color_names) inherit properties from
    their spatially nearest labelled neighbour.

    Args:
        labels: (H, W) int array after napari editing.
        base_properties: Existing {color_name: props} mapping.
        color_names: List indexed by original label id.

    Returns:
        Extended property dict covering all non-zero labels.
    """
    unique = sorted(set(labels.flatten()) - {0})
    out: dict[str, dict] = {}

    for lid in unique:
        name = (
            color_names[lid - 1]
            if color_names and lid - 1 < len(color_names)
            else f"layer_{lid}"
        )
        if name in base_properties:
            out[name] = dict(base_properties[name])
        else:
            # Inherit from nearest neighbour
            mask = labels == lid
            dist, indices = ndimage.distance_transform_edt(
                ~mask & (labels != 0), return_indices=True
            )
            # Find the nearest non-zero, non-self pixel
            rr, cc = np.where(mask)
            if len(rr) == 0:
                continue
            # Sample a few pixels and take majority neighbour label
            samples = min(10, len(rr))
            idx = np.linspace(0, len(rr) - 1, samples, dtype=int)
            ny = indices[0][rr, cc][idx]
            nx = indices[1][rr, cc][idx]
            neighbour_labels = labels[ny, nx]
            valid = neighbour_labels > 0
            if not valid.any():
                # Fallback to first known property or generated
                fallback = next(iter(base_properties.values()), None)
                if fallback:
                    out[name] = dict(fallback)
                else:
                    out[name] = {"Vp": 3000.0, "Vs": 1732.0, "rho": 2200.0}
                continue
            parent_label = int(np.bincount(neighbour_labels[valid]).argmax())
            parent_name = (
                color_names[parent_label - 1]
                if color_names and parent_label - 1 < len(color_names)
                else f"layer_{parent_label}"
            )
            parent_props = base_properties.get(parent_name)
            if parent_props:
                out[name] = dict(parent_props)
            else:
                fallback = next(iter(base_properties.values()), None)
                if fallback:
                    out[name] = dict(fallback)
                else:
                    out[name] = {"Vp": 3000.0, "Vs": 1732.0, "rho": 2200.0}
    return out


__all__ = [
    "DEFAULT_PROPERTIES",
    "assign_properties",
    "load_properties_json",
    "save_properties_json",
    "build_properties_template",
]
