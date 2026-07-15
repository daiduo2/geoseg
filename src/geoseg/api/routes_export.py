"""Export API routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from geoseg.api.schemas import ExportSpecfemResponse

router = APIRouter(prefix="/api/export")


@router.post("/specfem", response_model=ExportSpecfemResponse)
async def export_specfem(
    labels: UploadFile = File(...),
    color_names: str = Form(...),
) -> ExportSpecfemResponse:
    """Export labels to SPECFEM tomography file + Par_file snippet."""
    import tempfile

    from geoseg.modules.exporter.specfem import (
        labels_to_grids,
        write_parfile_snippet,
        write_tomography_file,
    )
    from geoseg.modules.post_process.properties import (
        assign_properties,
        generate_properties_for_layers,
    )

    npz = np.load(labels.file)
    labels_arr: np.ndarray = npz["labels"].astype(np.int32)
    parsed_color_names: list[str] = json.loads(color_names)

    try:
        props = assign_properties(parsed_color_names)
    except ValueError:
        props = generate_properties_for_layers(parsed_color_names)

    vp, vs, rho = labels_to_grids(
        labels_arr,
        props,
        color_names=parsed_color_names,
    )

    h, w = labels_arr.shape
    x_coords = np.linspace(0, w - 1, w)
    z_coords = np.linspace(0, h - 1, h)

    with tempfile.TemporaryDirectory() as tmpdir:
        tomo_path = Path(tmpdir) / "tomo.xyz"
        parfile_path = Path(tmpdir) / "parfile_snippet.txt"

        write_tomography_file(vp, vs, rho, x_coords, z_coords, tomo_path)
        write_parfile_snippet(
            parsed_color_names,
            props,
            parfile_path,
            nx=w,
            nz=h,
        )

        tomo_content = tomo_path.read_text(encoding="utf-8")
        parfile_content = parfile_path.read_text(encoding="utf-8")

    return ExportSpecfemResponse(
        tomo_xyz=tomo_content,
        parfile_snippet=parfile_content,
    )


__all__ = ["export_specfem", "router"]
