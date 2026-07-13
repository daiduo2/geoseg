"""Editor state model (napari-independent).

Manages shapes, shape types, label recomputation, and one-time endpoint
snapping. All operations return new state or mutate only this object's own
fields; input arrays are treated as immutable.
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.editor.editor_core import (
    RegionProperties,
    shapes_to_labels,
    snap_line_endpoints,
    snap_path_endpoints,
)


class EditorModel:
    """Shapes-primary state machine for the segmentation editor.

    Keeps shapes and types aligned with napari's Shapes layer, but contains
    no Qt/napari dependencies so it can be unit-tested headlessly.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        properties: RegionProperties | None = None,
    ) -> None:
        self.image_shape = image_shape
        self.shapes: list[np.ndarray] = []
        self.shape_types: list[str] = []
        self.properties = properties if properties is not None else RegionProperties()
        # Indices of shapes that have already been processed by snap_shape.
        # Once a shape is in this set it will never be auto-snapped again,
        # even if the user drags its vertices.
        self._snapped_indices: set[int] = set()

    # -----------------------------------------------------------------------
    # Shape lifecycle
    # -----------------------------------------------------------------------

    def add_shape(self, vertices: np.ndarray, shape_type: str) -> int:
        """Append a shape and return its index.

        New shapes are eligible for one-time endpoint snapping.
        """
        self.shapes.append(np.asarray(vertices, dtype=float))
        self.shape_types.append(shape_type)
        return len(self.shapes) - 1

    def add_initial_shape(self, vertices: np.ndarray, shape_type: str) -> int:
        """Append a shape that is already considered "final" (e.g. from labels).

        Initial shapes are marked as snapped so they are not auto-adjusted later.
        """
        idx = self.add_shape(vertices, shape_type)
        self._snapped_indices.add(idx)
        return idx

    def update_shape(self, index: int, vertices: np.ndarray) -> None:
        """Replace the vertices of an existing shape."""
        if not (0 <= index < len(self.shapes)):
            raise IndexError(f"Shape index {index} out of range")
        self.shapes[index] = np.asarray(vertices, dtype=float)

    def remove_shape(self, index: int) -> None:
        """Remove a shape and shift higher indices down."""
        if not (0 <= index < len(self.shapes)):
            raise IndexError(f"Shape index {index} out of range")
        del self.shapes[index]
        del self.shape_types[index]
        self._snapped_indices.discard(index)
        self._snapped_indices = {
            j if j < index else j - 1 for j in self._snapped_indices
        }

    def get_shape(self, index: int) -> np.ndarray:
        """Return a copy of the shape vertices."""
        if not (0 <= index < len(self.shapes)):
            raise IndexError(f"Shape index {index} out of range")
        return np.asarray(self.shapes[index], dtype=float)

    # -----------------------------------------------------------------------
    # Topology
    # -----------------------------------------------------------------------

    def recompute_labels(self) -> np.ndarray:
        """Compute labels from current shapes.

        An empty shape list yields a single region (the whole canvas) rather
        than all zeros.
        """
        return shapes_to_labels(self.shapes, self.shape_types, self.image_shape)

    # -----------------------------------------------------------------------
    # Snapping
    # -----------------------------------------------------------------------

    def snap_shape(self, index: int, labels: np.ndarray) -> bool:
        """Snap one shape's endpoints to region boundaries.

        Args:
            index: Shape index in self.shapes.
            labels: (H, W) label array *before* this shape is applied.

        Returns:
            True if the shape was changed, False otherwise.
        """
        if not (0 <= index < len(self.shapes)):
            raise IndexError(f"Shape index {index} out of range")

        if index in self._snapped_indices:
            return False

        shape_type = self.shape_types[index]
        if shape_type not in ("line", "path"):
            self._snapped_indices.add(index)
            return False

        vertices = np.asarray(self.shapes[index], dtype=float)
        if len(vertices) < 2:
            self._snapped_indices.add(index)
            return False

        new_vertices = self._snap_vertices(vertices, shape_type, labels)
        self._snapped_indices.add(index)
        if new_vertices is None:
            return False

        self.shapes[index] = new_vertices
        return True

    def _snap_vertices(
        self,
        vertices: np.ndarray,
        shape_type: str,
        labels: np.ndarray,
    ) -> np.ndarray | None:
        """Return snapped vertices or None if no change."""
        h, w = labels.shape

        if shape_type == "line" and len(vertices) == 2:
            return self._snap_line(vertices, labels, h, w)

        if shape_type == "path":
            return self._snap_path(vertices, labels, h, w)

        return None

    def _snap_line(
        self,
        vertices: np.ndarray,
        labels: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray | None:
        p1 = (float(vertices[0, 1]), float(vertices[0, 0]))
        p2 = (float(vertices[1, 1]), float(vertices[1, 0]))

        mid_y = int(round((vertices[0, 0] + vertices[1, 0]) / 2))
        mid_x = int(round((vertices[0, 1] + vertices[1, 1]) / 2))
        if not (0 <= mid_y < h and 0 <= mid_x < w):
            return None

        target = labels[mid_y, mid_x]
        if target == 0:
            return None

        mask = labels == target
        trimmed = snap_line_endpoints(mask, p1, p2)
        if trimmed is None:
            return None

        b1, b2 = trimmed
        new_vertices = np.array([[b1[1], b1[0]], [b2[1], b2[0]]], dtype=float)

        if np.allclose(vertices, new_vertices, atol=1.0):
            return None
        return new_vertices

    def _snap_path(
        self,
        vertices: np.ndarray,
        labels: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray | None:
        mid_y = int(round((vertices[0, 0] + vertices[1, 0]) / 2))
        mid_x = int(round((vertices[0, 1] + vertices[1, 1]) / 2))
        if not (0 <= mid_y < h and 0 <= mid_x < w):
            return None
        target_start = labels[mid_y, mid_x]

        mid_y = int(round((vertices[-2, 0] + vertices[-1, 0]) / 2))
        mid_x = int(round((vertices[-2, 1] + vertices[-1, 1]) / 2))
        if not (0 <= mid_y < h and 0 <= mid_x < w):
            return None
        target_end = labels[mid_y, mid_x]

        if target_start == 0 and target_end == 0:
            return None

        v = np.array(vertices, dtype=float)
        snapped = False

        if target_start != 0:
            mask_start = labels == target_start
            result = snap_path_endpoints(mask_start, v)
            if result is not None:
                v = result
                snapped = True

        if target_end != 0 and target_end != target_start:
            mask_end = labels == target_end
            result = snap_path_endpoints(mask_end, v)
            if result is not None:
                v = result
                snapped = True

        if not snapped or np.allclose(vertices, v, atol=1.0):
            return None
        return v
