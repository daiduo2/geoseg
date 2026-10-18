"""Interactive segmentation view with drag-and-drop region assignment.

Visualizes figure + overlay + component polygons. Small regions are draggable
onto large (locked) regions to merge them.

Usage:
    >>> from PySide6.QtWidgets import QApplication
    >>> app = QApplication([])
    >>> view = SegmentationView()
    >>> view.load_from_arrays(img_rgb, labels)
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
)
from skimage import measure
from skimage.measure import label as sk_label, regionprops


def _layer_color(layer_id: int) -> tuple[int, int, int]:
    """Deterministic color for a layer id."""
    rng = np.random.default_rng(layer_id * 42 + 7)
    return tuple(int(c) for c in rng.integers(80, 230, size=3))


def _extract_component_polygons(labels: np.ndarray) -> list[dict]:
    """Extract per-component contour polygons from a label map.

    Returns list of dicts:
        {
            "layer_id": int,
            "area": int,
            "centroid": [cx, cy],
            "points": [(x, y), ...],
        }
    """
    components: list[dict] = []
    for layer_id in sorted(set(labels.flatten()) - {0}):
        layer_mask = labels == layer_id
        labeled = sk_label(layer_mask)
        for rp in regionprops(labeled):
            comp_mask = labeled == rp.label
            contours = measure.find_contours(comp_mask.astype(np.uint8), level=0.5)
            if not contours:
                continue
            longest = max(contours, key=len)
            if len(longest) < 3:
                continue
            points = [(float(p[1]), float(p[0])) for p in longest]
            components.append({
                "layer_id": int(layer_id),
                "area": int(rp.area),
                "centroid": [float(rp.centroid[1]), float(rp.centroid[0])],
                "points": points,
            })
    return components


def _make_overlay_pixmap(labels: np.ndarray, alpha: int = 120) -> QPixmap:
    """Create a colored overlay pixmap from labels."""
    h, w = labels.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for lid in sorted(set(labels.flatten()) - {0}):
        color = _layer_color(lid)
        mask = labels == lid
        overlay[mask] = [*color, alpha]
    img = Image.fromarray(overlay, mode="RGBA")
    qimg = QImage(img.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


class RegionItem(QGraphicsPolygonItem):
    """A polygon region representing one connected component.

    Small regions are draggable; large regions act as drop targets.
    """

    def __init__(
        self,
        layer_id: int,
        area: int,
        centroid: tuple[float, float],
        points: list[tuple[float, float]],
        is_small: bool,
        parent: "SegmentationView" | None = None,
    ):
        poly = QPolygonF([QPointF(x, y) for x, y in points])
        super().__init__(poly)
        self.layer_id = layer_id
        self.area = area
        self.centroid = centroid
        self.is_small = is_small
        self._view = parent

        color = QColor(*_layer_color(layer_id))
        self._normal_pen = QPen(
            QColor(255, 255, 255) if is_small else QColor(0, 0, 0),
            2 if is_small else 3,
        )
        self._selected_pen = QPen(QColor(255, 215, 0), 4)
        self.setPen(self._normal_pen)
        self.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
        self.setAcceptHoverEvents(True)

        if is_small:
            self.setFlags(
                QGraphicsPolygonItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsPolygonItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def hoverEnterEvent(self, event):
        if not self.is_small:
            self.setPen(QPen(QColor(0, 255, 0), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self.is_small:
            self.setPen(self._normal_pen)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if self.is_small and event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._view:
                self._view.select_region(self)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_small and event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if self._view:
                self._view.on_region_dropped(self)
        super().mouseReleaseEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.setPen(self._selected_pen if selected else self._normal_pen)


class SegmentationView(QGraphicsView):
    """Interactive view for manual region adjustment.

    Signals / public API:
        load_from_arrays(img_rgb, labels, area_threshold=0.02)
        get_labels() -> np.ndarray  # returns modified labels
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self._img_rgb: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._area_threshold: float = 0.02
        self._region_items: list[RegionItem] = []
        self._selected_item: RegionItem | None = None
        self._total_area: int = 0

    def clear(self) -> None:
        self._scene.clear()
        self._region_items.clear()
        self._selected_item = None
        self._img_rgb = None
        self._labels = None

    def load_from_arrays(
        self,
        img_rgb: np.ndarray,
        labels: np.ndarray,
        area_threshold: float = 0.02,
    ) -> None:
        """Load an image + label map and build interactive regions."""
        self.clear()
        self._img_rgb = img_rgb
        self._labels = labels.copy()
        self._area_threshold = area_threshold
        self._total_area = int((labels != 0).sum())

        h, w = img_rgb.shape[:2]

        # Background image
        rgb_img = Image.fromarray(img_rgb)
        qimg = QImage(rgb_img.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._scene.addItem(QGraphicsPixmapItem(pixmap))

        # Overlay
        overlay = _make_overlay_pixmap(labels)
        overlay_item = QGraphicsPixmapItem(overlay)
        overlay_item.setOpacity(0.6)
        self._scene.addItem(overlay_item)

        # Region polygons
        threshold_px = int(self._total_area * area_threshold)
        components = _extract_component_polygons(labels)
        for comp in components:
            is_small = comp["area"] < threshold_px
            item = RegionItem(
                layer_id=comp["layer_id"],
                area=comp["area"],
                centroid=tuple(comp["centroid"]),
                points=comp["points"],
                is_small=is_small,
                parent=self,
            )
            self._scene.addItem(item)
            self._region_items.append(item)

        self.setSceneRect(0, 0, w, h)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def select_region(self, item: RegionItem) -> None:
        if self._selected_item and self._selected_item != item:
            self._selected_item.set_selected(False)
        self._selected_item = item
        item.set_selected(True)

    def on_region_dropped(self, dragged: RegionItem) -> None:
        """Called when a small region is released. Merge it into the big region below."""
        if not self._labels is not None:
            return

        # Find the big region whose polygon contains the dragged item's centroid
        cx, cy = dragged.sceneBoundingRect().center().x(), dragged.sceneBoundingRect().center().y()
        target = None
        for item in self._region_items:
            if not item.is_small and item.contains(QPointF(cx, cy)):
                target = item
                break

        if target is None or target.layer_id == dragged.layer_id:
            # No target or same layer — snap back to original position
            dragged.setPos(0, 0)
            return

        # Merge: change dragged region's pixels to target's layer
        # We need to know which pixels belong to this specific component.
        # Since RegionItem doesn't store pixel mask, we use connected-component
        # matching by centroid proximity.
        self._merge_by_centroid(dragged, target, cx, cy)

        # Refresh view on next event loop so mouseReleaseEvent can finish
        # before the scene (and the dragged C++ object) is destroyed.
        QTimer.singleShot(0, lambda: self.load_from_arrays(
            self._img_rgb, self._labels, self._area_threshold
        ))

    def _merge_by_centroid(self, dragged: RegionItem, target: RegionItem, cx: float, cy: float) -> None:
        """Find the component near release position and relabel it to target layer."""
        labels = self._labels

        # Find connected component in dragged.layer_id closest to release pos
        layer_mask = labels == dragged.layer_id
        labeled = sk_label(layer_mask)
        best_label = 0
        best_dist = float("inf")
        for rp in regionprops(labeled):
            dist = (rp.centroid[1] - cx) ** 2 + (rp.centroid[0] - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_label = rp.label

        if best_label > 0:
            comp_mask = labeled == best_label
            labels[comp_mask] = target.layer_id

    def get_labels(self) -> np.ndarray | None:
        return self._labels.copy() if self._labels is not None else None

    def contextMenuEvent(self, event) -> None:
        item = self._scene.itemAt(self.mapToScene(event.pos()), self.transform())
        if isinstance(item, RegionItem) and item.is_small:
            menu = QMenu(self)
            # Build actions: assign to each big region layer
            big_layers = {ri.layer_id for ri in self._region_items if not ri.is_small}
            for lid in sorted(big_layers):
                act = menu.addAction(f"Assign to layer {lid}")
                act.triggered.connect(lambda _=None, l=lid: self._assign_layer(item, l))
            menu.exec(event.globalPos())
        else:
            super().contextMenuEvent(event)

    def _assign_layer(self, item: RegionItem, new_layer_id: int) -> None:
        if self._labels is None:
            return
        # Find component by centroid and relabel
        labels = self._labels
        layer_mask = labels == item.layer_id
        labeled = sk_label(layer_mask)
        cx, cy = item.centroid
        best_label = 0
        best_dist = float("inf")
        for rp in regionprops(labeled):
            dist = (rp.centroid[1] - cx) ** 2 + (rp.centroid[0] - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_label = rp.label
        if best_label > 0:
            labels[labeled == best_label] = new_layer_id
        self.load_from_arrays(self._img_rgb, labels, self._area_threshold)
