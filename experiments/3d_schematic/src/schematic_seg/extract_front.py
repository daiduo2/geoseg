"""Extract front faces from 3D oblique schematic panels.

Strategy:
1. Binarize each panel (white bg vs colored cube)
2. Extract outer contour of the cube
3. Approximate contour as polygon
4. Identify bottom edge (lowest) and left edge (leftmost)
5. Compute parallelogram corners from these two edges
6. Affine warp to uniform rectangle
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def find_polygon_edges(content: np.ndarray) -> tuple | None:
    """Find bottom and left edges of front face from binary mask contour."""
    gray = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    # Close small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    # Approximate as polygon
    peri = cv2.arcLength(largest, True)
    poly = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(poly) < 4:
        # Try convex hull if approximation is too simple
        poly = cv2.convexHull(largest)
        if len(poly) < 4:
            return None

    # Extract edges from polygon vertices
    pts = poly.reshape(-1, 2).astype(float)
    n = len(pts)

    edges = []
    for i in range(n):
        j = (i + 1) % n
        p1, p2 = pts[i], pts[j]
        mid = (p1 + p2) / 2
        length = np.linalg.norm(p2 - p1)
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
        edges.append((p1, p2, length, angle, mid))

    # Bottom edge: edge with largest mid_y (lowest)
    # But it should also be reasonably long
    bottom = max(edges, key=lambda e: e[4][1])
    # Left edge: edge with smallest mid_x (leftmost)
    left = min(edges, key=lambda e: e[4][0])

    return bottom, left


def compute_parallelogram(bottom_edge: tuple, left_edge: tuple) -> np.ndarray | None:
    """Compute parallelogram corners from bottom and left edges."""
    bp1, bp2, blen, bangle, _ = bottom_edge
    lp1, lp2, llen, langle, _ = left_edge

    # Find intersection of the two lines (bottom-left corner)
    x1, y1 = bp1
    x2, y2 = bp2
    x3, y3 = lp1
    x4, y4 = lp2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    A = np.array([px, py], dtype=float)

    # Bottom-right: endpoint of bottom edge farthest from A
    d1 = np.linalg.norm(bp1 - A)
    d2 = np.linalg.norm(bp2 - A)
    B = bp1.copy() if d1 > d2 else bp2.copy()

    # Top-left: endpoint of left edge farthest from A
    d1 = np.linalg.norm(lp1 - A)
    d2 = np.linalg.norm(lp2 - A)
    D = lp1.copy() if d1 > d2 else lp2.copy()

    # Top-right: D + (B - A) for parallelogram
    C = D + (B - A)

    return np.array([D, C, B, A], dtype=np.float32)


def warp_front_face(content: np.ndarray, quad: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Affine warp parallelogram to rectangle."""
    D, C, B, A = quad
    src = np.array([D, C, A], dtype=np.float32)
    dst = np.array([[0, 0], [target_w, 0], [0, target_h]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(content, M, (target_w, target_h))


def extract_panels(image_path: str | Path, panel_ranges: list[tuple[int, int]]) -> list[np.ndarray]:
    """Extract front faces from all panels."""
    img = np.array(Image.open(image_path).convert("RGB"))
    results = []

    for s, e in panel_ranges:
        panel = img[:, s:e]
        content = panel[90:, :]  # Skip title

        edges = find_polygon_edges(content)
        if edges is None:
            results.append(None)
            continue

        bottom, left = edges
        quad = compute_parallelogram(bottom, left)
        if quad is None:
            results.append(None)
            continue

        results.append((content, quad))

    return results


def main():
    img_path = Path(__file__).parent.parent.parent / "figures" / "panels" / "panel_1.png"
    panel_ranges = [(0, 395), (432, 828), (861, 1269)]

    extracted = extract_panels(img_path, panel_ranges)

    # Determine standard size from panel 1 + 1/3 height
    if extracted[0] is not None:
        _, quad0 = extracted[0]
        D, C, B, A = quad0
        h0 = int(np.linalg.norm(D - A))
        w0 = int(np.linalg.norm(B - A))
        target_h = int(h0 * 4 / 3)
        target_w = int(w0 * h0 * 4 / 3 / h0)  # Keep aspect ratio
        target_w = int(w0)
    else:
        target_w, target_h = 300, 550

    print(f"Target size: {target_w}x{target_h}")

    for i, result in enumerate(extracted):
        if result is None:
            print(f"Panel {i + 1}: failed")
            continue
        content, quad = result
        warped = warp_front_face(content, quad, target_w, target_h)
        out_path = Path(__file__).parent.parent.parent / "figures" / "panels" / f"panel_{i + 1}.png"
        Image.fromarray(warped).save(out_path)
        print(f"Panel {i + 1}: saved {out_path} ({warped.shape[1]}x{warped.shape[0]})")


if __name__ == "__main__":
    main()
