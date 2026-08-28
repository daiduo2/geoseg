"""Rasterize GeoJSON polygons onto a regular Cartesian grid.

Public API:
    rasterize_polygons(features, color_names, dx, dz, x_range, z_range)
        -> GridResult
    label_grid_to_properties(grid, color_names, props)
        -> (vp_grid, vs_grid, rho_grid)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.draw import polygon as draw_polygon


@dataclass
class GridResult:
    """Output of rasterization."""
    labels: np.ndarray          # (nz, nx) int, values 0..k-1 (or -1 = undefined)
    x_coords: np.ndarray        # (nx,) physical x in km
    z_coords: np.ndarray        # (nz,) physical z (depth, positive downward) in km
    dx: float
    dz: float
    color_names: list[str]


def _polygon_mask(
    coords: list[list[float]],
    shape: tuple[int, int],
    x_min: float,
    z_min: float,
    dx: float,
    dz: float,
) -> np.ndarray:
    """Return a boolean mask for one polygon ring.

    coords: list of [x, z] in physical units.
    shape: (nz, nx)
    """
    xs = np.array([p[0] for p in coords])
    zs = np.array([p[1] for p in coords])
    # Convert physical -> pixel
    rr = ((zs - z_min) / dz).astype(int)
    cc = ((xs - x_min) / dx).astype(int)
    # skimage.draw.polygon expects (row, col) = (z, x)
    rr, cc = draw_polygon(rr, cc, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def rasterize_polygons(
    features: dict,
    color_names: list[str],
    dx: float = 1.0,
    dz: float = 1.0,
    x_range: tuple[float, float] = (0.0, 100.0),
    z_range: tuple[float, float] = (0.0, 50.0),
) -> GridResult:
    """Rasterize a GeoJSON FeatureCollection onto a regular grid.

    `features`: {"type": "FeatureCollection", "features": [...]}
    `dx`, `dz`: grid spacing in km.
    `x_range`: (x_min, x_max) in km.
    `z_range`: (z_min, z_max) in km; z positive downward (depth).

    Polygons are filled in input order; later polygons overwrite earlier ones
    within the same color zone.  Different color zones should not overlap in
    a well-segmented figure, but if they do the last one wins.
    """
    x_min, x_max = x_range
    z_min, z_max = z_range
    nx = int(np.ceil((x_max - x_min) / dx)) + 1
    nz = int(np.ceil((z_max - z_min) / dz)) + 1

    labels = np.full((nz, nx), -1, dtype=np.int32)
    x_coords = np.linspace(x_min, x_max, nx)
    z_coords = np.linspace(z_min, z_max, nz)

    color_to_idx = {name: i for i, name in enumerate(color_names)}

    for feat in features.get("features", []):
        props = feat.get("properties", {})
        name = props.get("color_name")
        if name not in color_to_idx:
            continue
        idx = color_to_idx[name]
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        # GeoJSON Polygon: coordinates[0] is the outer ring
        ring = geom["coordinates"][0]
        mask = _polygon_mask(ring, (nz, nx), x_min, z_min, dx, dz)
        labels[mask] = idx

    return GridResult(
        labels=labels,
        x_coords=x_coords,
        z_coords=z_coords,
        dx=dx,
        dz=dz,
        color_names=color_names,
    )


def label_grid_to_properties(
    grid: np.ndarray,
    color_names: list[str],
    props: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert label grid to Vp, Vs, rho grids.

    `grid`: (nz, nx) int labels.
    `props`: {color_name: {"Vp": float, "Vs": float, "rho": float}}

    Returns (vp, vs, rho) each (nz, nx).  Undefined pixels (-1) are set to
    the mean of all defined values as a soft fallback.
    """
    nz, nx = grid.shape
    vp = np.zeros((nz, nx), dtype=np.float64)
    vs = np.zeros((nz, nx), dtype=np.float64)
    rho = np.zeros((nz, nx), dtype=np.float64)

    for idx, name in enumerate(color_names):
        mask = grid == idx
        if not mask.any():
            continue
        p = props[name]
        vp[mask] = p["Vp"]
        vs[mask] = p["Vs"]
        rho[mask] = p["rho"]

    # Fallback for undefined pixels: nearest-neighbor fill
    undefined = grid == -1
    if undefined.any():
        defined_vp = vp[~undefined]
        if defined_vp.size > 0:
            vp[undefined] = defined_vp.mean()
            vs[undefined] = vs[~undefined].mean()
            rho[undefined] = rho[~undefined].mean()

    return vp, vs, rho


__all__ = ["GridResult", "rasterize_polygons", "label_grid_to_properties"]
