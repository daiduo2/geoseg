"""SPECFEM2D/3D model output writers.

Bridges post-processed labels + physical properties to SPECFEM-compatible
model files.

Test scenario:
    >>> import numpy as np
    >>> labels = np.array([[1,1,2,2],[1,1,2,2],[3,3,3,0],[3,3,3,0]])
    >>> props = {"layer_1": {"Vp": 5000.0, "Vs": 2887.0, "rho": 2600.0},
    ...           "layer_2": {"Vp": 4000.0, "Vs": 2309.0, "rho": 2400.0},
    ...           "layer_3": {"Vp": 3000.0, "Vs": 1732.0, "rho": 2200.0}}
    >>> vp, vs, rho = labels_to_grids(labels, props)
    >>> assert vp.shape == labels.shape
    >>> assert vp[0,0] == 5000.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def labels_to_grids(
    labels: np.ndarray,
    properties: dict[str, dict],
    color_names: list[str] | None = None,
    default_vp: float = 3000.0,
    default_vs: float = 1732.0,
    default_rho: float = 2200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a label map + property dict to Vp / Vs / rho grids.

    Args:
        labels: (H, W) int array. 0 = background (filled with defaults).
        properties: {color_name: {"Vp": float, "Vs": float, "rho": float}}.
        color_names: Optional list indexed by label id (label 0 ignored).
            If None, keys are looked up as "layer_{label_id}".
        default_vp/vs/rho: Background fill values for label 0.

    Returns:
        (vp, vs, rho) arrays of shape (H, W).
    """
    h, w = labels.shape
    vp = np.full((h, w), default_vp, dtype=np.float64)
    vs = np.full((h, w), default_vs, dtype=np.float64)
    rho = np.full((h, w), default_rho, dtype=np.float64)

    for label_id in sorted(set(labels.flatten()) - {0}):
        name = (
            color_names[label_id - 1]
            if color_names and label_id - 1 < len(color_names)
            else f"layer_{label_id}"
        )
        if name not in properties:
            continue
        mask = labels == label_id
        p = properties[name]
        vp[mask] = float(p.get("Vp", default_vp))
        vs[mask] = float(p.get("Vs", default_vs))
        rho[mask] = float(p.get("rho", default_rho))

    return vp, vs, rho


def write_tomography_file(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    path: str | Path,
    include_attenuation: bool = False,
    qp: np.ndarray | None = None,
    qs: np.ndarray | None = None,
) -> None:
    """Write SPECFEM2D `tomography_file` (MODEL = tomo).

    Format (per SPECFEM2D manual):
        #x #z #vp #vs #rho [#Qp #Qs]
        x1 z1 vp1 vs1 rho1 [Qp1 Qs1]
        ...

    Arrays are assumed (nz, nx). Output is written z-major (row-major)
    so that x changes fastest — compatible with SPECFEM2D internal reader.
    """
    path = Path(path)
    nz, nx = vp.shape
    with path.open("w", encoding="utf-8") as f:
        header = "#x #z #Vp #Vs #rho"
        if include_attenuation:
            header += " #Qp #Qs"
        f.write(header + "\n")
        for iz in range(nz):
            z = z_coords[iz]
            for ix in range(nx):
                line = f"{x_coords[ix]:.6f} {z:.6f} {vp[iz, ix]:.4f} {vs[iz, ix]:.4f} {rho[iz, ix]:.4f}"
                if include_attenuation and qp is not None and qs is not None:
                    line += f" {qp[iz, ix]:.4f} {qs[iz, ix]:.4f}"
                f.write(line + "\n")


def write_parfile_snippet(
    color_names: list[str],
    props: dict[str, dict],
    path: str | Path,
    nx: int = 100,
    nz: int = 50,
    dx: float = 1.0,
    dz: float = 1.0,
) -> None:
    """Emit a commented snippet for SPECFEM2D Par_file.

    The user still needs to merge this into their full Par_file manually.
    """
    lines = [
        "# --- geo-segment generated snippet ---",
        "# Place these lines into your SPECFEM2D Par_file",
        "",
        "MODEL                           = tomo",
        "TOMOGRAPHY_FILE                 = ./DATA/tomo_file.xyz",
        "",
        "# Model dimensions (must match tomography_file grid)",
        f"nx_tomo                         = {nx}",
        f"nz_tomo                         = {nz}",
        f"dx_tomo                         = {dx:.6f}",
        f"dz_tomo                         = {dz:.6f}",
        "",
        "# Color zone -> property mapping (for reference)",
    ]
    for name in color_names:
        p = props.get(name, {})
        lines.append(
            f"#   {name:10s}:  Vp={p.get('Vp', '?'):.1f}  "
            f"Vs={p.get('Vs', '?'):.1f}  rho={p.get('rho', '?'):.1f}"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_external_model_ascii(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    path: str | Path,
) -> None:
    """Write a simple ASCII external model file.

    Some SPECFEM2D versions accept an external ASCII model with header:
        nx nz
        x1 x2 ... xnx
        z1 z2 ... znz
        vp(1,1) vp(1,2) ...
        ...
    This is less common than tomography_file; provided for compatibility.
    """
    path = Path(path)
    nz, nx = vp.shape
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{nx} {nz}\n")
        f.write(" ".join(f"{x:.6f}" for x in x_coords) + "\n")
        f.write(" ".join(f"{z:.6f}" for z in z_coords) + "\n")
        for iz in range(nz):
            f.write(" ".join(f"{vp[iz, ix]:.4f}" for ix in range(nx)) + "\n")
        for iz in range(nz):
            f.write(" ".join(f"{vs[iz, ix]:.4f}" for ix in range(nx)) + "\n")
        for iz in range(nz):
            f.write(" ".join(f"{rho[iz, ix]:.4f}" for ix in range(nx)) + "\n")


__all__ = [
    "labels_to_grids",
    "write_tomography_file",
    "write_parfile_snippet",
    "write_external_model_ascii",
]
