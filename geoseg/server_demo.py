"""Week 1 validation: test server endpoints without starting HTTP server.

Uses FastAPI TestClient to exercise endpoints directly.
Run:
    python -m geoseg.server_demo
"""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

try:
    from fastapi.testclient import TestClient
except ImportError as exc:
    raise ImportError(
        "fastapi testclient not available. Install with: pip install httpx"
    ) from exc

from geoseg.server import app

client = TestClient(app)


def _make_image_file(
    shape: tuple[int, int] = (200, 300), color: tuple[int, int, int] = (128, 128, 128)
) -> io.BytesIO:
    """Create a fake PNG image in memory."""
    arr = np.full((*shape, 3), color, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_detect_panels() -> None:
    """M7 acceptance: detect-panels returns list[PanelInput]."""
    img = _make_image_file()
    resp = client.post("/api/agent/detect-panels", files={"image": ("test.png", img, "image/png")})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    for p in data:
        assert "id" in p
        assert "bbox" in p
        assert len(p["bbox"]) == 4
        assert "source" in p
    print(f"✓ detect_panels: {len(data)} panel(s) found")


def test_segment_agent() -> None:
    """M7 acceptance: segment returns SegmentationResult with contours."""
    img = _make_image_file()
    resp = client.post(
        "/api/agent/segment",
        files={"image": ("test.png", img, "image/png")},
        data={"n_layers": 3},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "contours" in data
    assert "meta" in data
    assert data["meta"]["engine"]
    print(f"✓ segment_agent: engine={data['meta']['engine']}, contours={len(data['contours'])}")


def test_export_specfem() -> None:
    """M7 acceptance: export returns SPECFEM files."""
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 0], [2, 2, 2, 0]], dtype=np.int32)
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=labels)
    buf.seek(0)

    resp = client.post(
        "/api/export/specfem",
        files={"labels": ("labels.npz", buf, "application/octet-stream")},
        data={"color_names": json.dumps(["layer_1", "layer_2"])},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "tomo_xyz" in data
    assert "parfile_snippet" in data
    assert "#x #z #Vp #Vs #rho" in data["tomo_xyz"]
    print(f"✓ export_specfem: tomo={len(data['tomo_xyz'])} chars, parfile={len(data['parfile_snippet'])} chars")


if __name__ == "__main__":
    test_detect_panels()
    test_segment_agent()
    test_export_specfem()
    print("\n✓ All Week 1 server endpoint tests passed.")
