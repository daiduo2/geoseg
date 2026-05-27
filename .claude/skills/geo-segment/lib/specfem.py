"""SPECFEM2D/3D model output writers.

Public API:
    write_tomography_file(vp, vs, rho, x_coords, z_coords, path)
        Write SPECFEM2D-compatible tomography file (x z Vp Vs rho).

    write_parfile_snippet(color_names, props, path)
        Emit a commented Par_file snippet showing MODEL=tomo setup.

    write_external_model_ascii(vp, vs, rho, x_coords, z_coords, path)
        Alternative: external_model format (header + binary-like rows).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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

    Arrays are assumed (nz, nx).  Output is written z-major (row-major)
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
        f"MODEL                           = tomo",
        f"TOMOGRAPHY_FILE                 = ./DATA/tomo_file.xyz",
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
    "write_tomography_file",
    "write_parfile_snippet",
    "write_external_model_ascii",
]
