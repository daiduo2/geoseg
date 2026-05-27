"""Map color zones to elastic properties (Vp, Vs, rho).

The default table is a coarse crustal-scale template based on typical
continental velocity structures.  Users override per-study via JSON.

Public API:
    assign_properties(color_names, custom_map=None) -> dict[str, dict]
    load_properties_json(path) -> dict[str, dict]
    save_properties_json(props, path)
"""

from __future__ import annotations

import json
from pathlib import Path


# Crustal-scale template — continental, no units enforced here
# (typical units: Vp/Vs in m/s or km/s; rho in kg/m³ or g/cm³)
DEFAULT_PROPERTIES = {
    "red":    {"Vp": 6500.0, "Vs": 3750.0, "rho": 2800.0},
    "orange": {"Vp": 5500.0, "Vs": 3200.0, "rho": 2700.0},
    "yellow": {"Vp": 4500.0, "Vs": 2600.0, "rho": 2500.0},
    "green":  {"Vp": 3500.0, "Vs": 2000.0, "rho": 2300.0},
    "blue":   {"Vp": 2500.0, "Vs": 1500.0, "rho": 2100.0},
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
                f"Provide it via --properties-json."
            )
    return out


def load_properties_json(path: str | Path) -> dict[str, dict]:
    """Load a user-supplied property table.

    Expected schema:
        {"red": {"Vp": 6500, "Vs": 3750, "rho": 2800}, ...}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Light validation
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


__all__ = [
    "DEFAULT_PROPERTIES",
    "assign_properties",
    "load_properties_json",
    "save_properties_json",
    "build_properties_template",
]
