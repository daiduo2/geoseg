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


__all__ = [
    "DEFAULT_PROPERTIES",
    "assign_properties",
    "load_properties_json",
    "save_properties_json",
    "build_properties_template",
]
