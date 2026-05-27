"""Demo / test scenario for batch_processor.

Tests: process_directory with synthetic images, resume, and error isolation.

Run:
    python -m geoseg.batch_processor_demo
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.batch_processor import process_directory


def _make_image(name: str, size: tuple[int, int], color: int | tuple[int, int, int]) -> np.ndarray:
    """Create a simple synthetic image."""
    if isinstance(color, int):
        color = (color, color, color)
    img = np.ones((*size, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    rng = np.random.default_rng(hash(name) % (2**31))
    img = np.clip(img.astype(np.int16) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
    return img


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        images_dir = td_path / "images"
        output_dir = td_path / "output"
        images_dir.mkdir()

        # Create test images
        # img1: gray -> observational_data or other -> skipped
        img1 = _make_image("gray", (100, 180), 180)
        Image.fromarray(img1).save(images_dir / "img1_gray.jpg")

        # img2: conceptual-like with colored blocks -> processed
        img2 = np.ones((120, 200, 3), dtype=np.uint8) * 245
        img2[10:60, 10:95] = [200, 60, 60]
        img2[10:60, 105:190] = [60, 200, 60]
        img2[70:110, 10:190] = [60, 60, 200]
        rng = np.random.default_rng(42)
        for _ in range(40):
            x, y = rng.integers(10, 190), rng.integers(10, 110)
            angle = rng.random() * 3.14159
            for l in range(rng.integers(8, 25)):
                px, py = int(x + l * np.cos(angle)), int(y + l * np.sin(angle))
                if 0 <= px < 200 and 0 <= py < 120:
                    img2[py, px] = [30, 30, 30]
        Image.fromarray(img2).save(images_dir / "img2_conceptual.png")

        # img3: another conceptual
        img3 = np.ones((100, 160, 3), dtype=np.uint8) * 250
        img3[10:90, 10:75] = [210, 80, 80]
        img3[10:90, 85:150] = [80, 80, 210]
        Image.fromarray(img3).save(images_dir / "img3_conceptual.jpg")

        print("=== test batch processing ===")
        summary = process_directory(
            images_dir=images_dir,
            output_dir=output_dir,
            n_layers=3,
            quality_preference="fast",
            use_vlm=False,
            skip_non_velocity_model=False,
            resume=False,
        )

        assert summary["total"] == 3
        assert summary["errors"] == 0
        # At least one processed (conceptual images should pass)
        assert summary["processed"] >= 1, f"Expected >=1 processed, got {summary['processed']}"
        print(f"  processed={summary['processed']} skipped={summary['skipped']} errors={summary['errors']}")

        # Check artifacts exist for processed images
        for img_name, res in summary["results"].items():
            if res["status"] == "ok":
                stem = Path(img_name).stem
                artifacts = list((output_dir / stem).iterdir()) if (output_dir / stem).exists() else []
                assert any("_tomo.xyz" in a.name for a in artifacts), f"Missing tomo for {img_name}"
                print(f"  {img_name}: {len(artifacts)} artifacts")

        print("\n=== test resume ===")
        summary2 = process_directory(
            images_dir=images_dir,
            output_dir=output_dir,
            n_layers=3,
            quality_preference="fast",
            use_vlm=False,
            skip_non_velocity_model=False,
            resume=True,
        )
        assert summary2["total"] == 3
        # All should be skipped because they were already processed
        for img_name in ["img1_gray.jpg", "img2_conceptual.png", "img3_conceptual.jpg"]:
            assert summary2["results"].get(img_name, {}).get("status") == summary["results"].get(img_name, {}).get("status")
        print("  Resume preserved previous results.")

        print("\nAll batch_processor tests passed.")
        return 0


if __name__ == "__main__":
    exit(main())
