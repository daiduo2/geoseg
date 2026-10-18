"""Procedural 2D conceptual-model panel generator.

Output contract (returned by `generate_panel`):

  panel_rgb : uint8 ndarray (H, W, 3) — the rendered RGB image.
  gt_labels : int32 ndarray (H, W)   — one label per pixel, values in 1..K.
              Background (above all layers) is 0.
  meta      : dict with keys:
                - "n_layers", "colormap"
                - "components": list of {kind, params} the panel was built from
                - "label_to_value": {label_id: float in [0, 1]} so an
                  evaluator can recover the original scalar field used
                  for rendering.

The generator is intentionally simple and deterministic per seed. It
is a 2D simplification of the BSM generator (Wu et al., Geophysics
2020) — same four component families (layers / faults / channels /
unconformities), no 3D, no folding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .colormaps import map_array  # noqa: PLC0415
from .configs import PanelConfig  # noqa: PLC0415


# Scalar-field value per layer. Equally spaced in [0, 1] before
# colormap mapping; layer 1 is the shallowest (smallest value).
def _layer_value(idx: int, n_layers: int) -> float:
    if n_layers <= 1:
        return 0.5
    # Top layer ~ 0.05, bottom layer ~ 0.95 — leave headroom at both ends
    return 0.05 + (0.9 * idx / (n_layers - 1))


def _layer_y_top(bottom: int, thickness: int) -> int:
    return max(0, bottom - thickness)


def _build_layer_y(
    h: int,
    n_layers: int,
    dip_deg: float,
    thickness_jitter: float,
    warp_amplitude: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return list of (y_top, y_bottom) for each layer, top to bottom.

    Layer 0 is the top layer; layer n_layers-1 is the bottom layer.
    """
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2 for a layered panel")

    # Mean layer thickness, with ±jitter
    mean_thick = h / n_layers
    thicknesses = np.array(
        [
            max(8, int(round(mean_thick * (1.0 + thickness_jitter * (2 * rng.random() - 1)))))
            for _ in range(n_layers)
        ],
        dtype=np.int64,
    )
    # Rescale so the stack fits exactly
    thicknesses = np.maximum(8, thicknesses * h // int(thicknesses.sum()))

    # Layer bottoms, top to bottom. Layer 0 bottom = first thickness.
    bottoms: list[int] = []
    y = 0
    for t in thicknesses:
        y = min(h, y + int(t))
        bottoms.append(y)
    bottoms[-1] = h  # force bottom to fill the panel

    # Apply dip: shift each layer's bottom by dip * (x_frac).
    # dip_deg in degrees; positive = layers slope down to the right.
    if abs(dip_deg) > 1e-3:
        shift_per_pixel = float(np.tan(np.deg2rad(dip_deg)))
        # total shift over width
        total_shift = shift_per_pixel * h  # approx; keep simple
        bottoms_with_dip: list[int] = []
        for i, b in enumerate(bottoms):
            # Earlier (top) layers shift more (proportional to depth from top)
            offset = int(round(total_shift * (i + 1) / n_layers))
            bottoms_with_dip.append(min(h, b + offset))
        bottoms = bottoms_with_dip

    # Warp: per-column vertical wobble for the bottom boundary of each
    # layer. The wobble itself is applied in `_render_scalar_field`; this
    # loop only preserves the original two RNG draws per layer so generated
    # panels stay reproducible.
    if warp_amplitude > 0:
        for _ in range(n_layers):
            rng.random()
            rng.uniform(1.0, 3.0)

    pairs: list[tuple[int, int]] = []
    prev_bottom = 0
    for b in bottoms:
        pairs.append((prev_bottom, int(b)))
        prev_bottom = int(b)
    return pairs


def _render_scalar_field(
    h: int,
    w: int,
    layer_pairs: list[tuple[int, int]],
    warp_amplitude: float,
    dip_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a (H, W) scalar field in [0, 1] from layer geometry.

    Adds mild per-row wobble to each layer boundary so the segmentation
    task isn't trivial (no perfectly horizontal lines).
    """
    n_layers = len(layer_pairs)
    field = np.zeros((h, w), dtype=np.float32)

    # Per-layer boundary y-position as a function of column x
    # Single shared phase/freq for all layers keeps neighbouring
    # boundaries in lockstep so wobble doesn't leave gap rows between
    # layers (BUG3 in audit). Amplitude scales per layer so deeper
    # layers still wobble more.
    if warp_amplitude > 0:
        shared_phase = rng.random() * 2 * np.pi
        shared_freq = rng.uniform(1.0, 3.0)
        shared_wobble = np.sin(np.linspace(0, shared_freq * np.pi, w) + shared_phase)
    for i, (yt, yb) in enumerate(layer_pairs):
        value = _layer_value(i, n_layers)
        if warp_amplitude > 0:
            amp = warp_amplitude * (i + 1) / n_layers
            wobble = amp * shared_wobble
            yt_w = yt + wobble
            yb_w = yb + wobble
        else:
            yt_w = np.full(w, yt, dtype=np.float32)
            yb_w = np.full(w, yb, dtype=np.float32)
        # NOTE: dip is already applied in `_build_layer_y` (bottoms
        # shifted there). Applying it again here would double the dip
        # and leave gap rows between layers — see BUG1 in audit.

        # Fill: for each column, fill rows in [yt_w[x], yb_w[x]] with `value`
        ys = np.arange(h, dtype=np.float32).reshape(-1, 1)
        in_layer = (ys >= yt_w.reshape(1, -1)) & (ys < yb_w.reshape(1, -1))
        field[in_layer] = value

    return field


def _apply_fault(
    field: np.ndarray,
    gt_labels: np.ndarray,
    fault_x_frac: float,
    offset: int,
) -> None:
    """Mutate field and gt_labels in place: introduce a vertical fault.

    Layers on the right of the fault are shifted downward by `offset`
    pixels (relative to the left side). Background label 0 is left alone.
    """
    h, w = field.shape
    x_fault = int(round(w * fault_x_frac))

    # Shift each row's column-content for columns to the right of fault.
    # We avoid numpy fancy indexing tricks; use a simple roll on the
    # field and a matching gt_labels.
    right_field = field[:, x_fault:]
    right_labels = gt_labels[:, x_fault:]
    shifted_field = np.zeros_like(right_field)
    shifted_labels = np.zeros_like(right_labels)
    if offset >= 0:
        if offset < h:
            shifted_field[offset:] = right_field[: h - offset]
            shifted_labels[offset:] = right_labels[: h - offset]
        # rows 0..offset remain 0 (gap appears)
    else:
        pos = -offset
        if pos < h:
            shifted_field[: h - pos] = right_field[pos:]
            shifted_labels[: h - pos] = right_labels[pos:]

    field[:, x_fault:] = shifted_field
    gt_labels[:, x_fault:] = shifted_labels


def _apply_channel(
    field: np.ndarray,
    gt_labels: np.ndarray,
    y_centre_frac: float,
    amplitude: float,
    thickness: int,
    rng: np.random.Generator,
) -> None:
    """Mutate field and gt_labels: carve a meandering channel.

    The channel is a new label (n_layers + 1) layered on top of the
    existing scalar field. We use a cosine + noise meander.
    """
    h, w = field.shape
    y_centre = h * y_centre_frac

    # Meander: low-frequency cosine + small per-column jitter
    base = amplitude * np.sin(np.linspace(0, 2 * np.pi, w))
    jitter = (rng.random(w) - 0.5) * amplitude * 0.3
    y_meander = y_centre + base + jitter

    new_label = int(gt_labels.max()) + 1
    ys = np.arange(h, dtype=np.float32).reshape(-1, 1)
    distance = np.abs(ys - y_meander.reshape(1, -1))
    in_channel = distance < (thickness / 2)
    field[in_channel] = 0.5  # arbitrary mid value for the channel fill
    gt_labels[in_channel] = new_label


def _apply_unconformity(
    field: np.ndarray,
    gt_labels: np.ndarray,
    y_frac: float,
    dip_below: float,
    dip_above: float,
    rng: np.random.Generator,
) -> None:
    """Mutate field and gt_labels: insert a horizontal unconformity.

    Layers above y_frac * h are re-labelled as new layer IDs (flat,
    unconformity surface); layers below keep their original dip but
    get new label IDs starting from n+1.
    """
    h, w = field.shape
    y_cut = int(round(h * y_frac))

    # We need the original unique labels and field values before mutation.
    # Use the field's value-quantised form: round to 0.1 to recover layer
    # grouping isn't trivial. Instead, regenerate above/below using the
    # current scalar field as a guide.
    above_mask = np.zeros((h, w), dtype=bool)
    below_mask = np.zeros((h, w), dtype=bool)
    # ys shape (h,) so scalar comparisons broadcast cleanly to (h, w)
    ys = np.arange(h, dtype=np.float32)

    # Above: flat layers from y_cut downward by ~equal thicknesses
    # (dip_above currently unused — kept in signature for future use)
    if dip_above != 0:
        shift = float(np.tan(np.deg2rad(dip_above))) * h
        xs = np.arange(w, dtype=np.float32)  # noqa: F841
    # Below: layers from y_cut down, each dipping
    if dip_below != 0:
        shift = float(np.tan(np.deg2rad(dip_below))) * h
    else:
        shift = 0.0

    n_above = 2
    above_thick = max(8, y_cut // n_above)
    # Last "above" layer extends down to y_cut (not strictly yb = (k+1)*thick
    # for the last layer), so above rows are fully covered.
    # Below: pick thickness first, then ceil(n) so the union covers
    # [y_cut, h) exactly — previously integer division left the last
    # row uncovered, retaining the old label there.
    below_thick = max(8, max(2, h // 100))  # ~100px per layer below
    n_below = max(2, -(-(h - y_cut) // below_thick))  # ceil division

    next_label = int(gt_labels.max()) + 1
    # Above: flat
    for k in range(n_above):
        yt = k * above_thick
        yb = (k + 1) * above_thick
        if yb > y_cut:
            yb = y_cut
        # ys[:,None] broadcasts (h,1) vs scalar -> (h,1), then .repeat(w,1) -> (h,w)
        m = (ys[:, None] >= yt) & (ys[:, None] < yb)
        m = np.broadcast_to(m, (h, w))
        above_mask |= m
        field[m] = 0.05 + 0.15 * (k + 1) / n_above
        gt_labels[m] = next_label
        next_label += 1
    # Below: dipping (each subsequent layer dips more)
    for k in range(n_below):
        base_yt = y_cut + k * below_thick
        base_yb = y_cut + (k + 1) * below_thick
        if base_yb > h:
            base_yb = h
        offset = shift * (k + 1) / n_below
        col_shift = offset * (np.arange(w, dtype=np.float32) / max(w - 1, 1))
        yt_w = base_yt + col_shift  # shape (w,)
        yb_w = base_yb + col_shift  # shape (w,)
        # (h, 1) vs (w,) -> (h, w) via broadcasting
        m = (ys[:, None] >= yt_w[None, :]) & (ys[:, None] < yb_w[None, :])
        below_mask |= m
        field[m] = 0.4 + 0.55 * (k + 1) / n_below
        gt_labels[m] = next_label
        next_label += 1

    # Re-assign background (rows above all layers): leave as 0.
    # Note: any pixels not covered by above/below still keep their old
    # label — should be none in practice since above+below = full height.


def generate_panel(cfg: PanelConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render a single synthetic panel.

    Returns (panel_rgb, gt_labels, meta).
    """
    rng = cfg.rng()
    h, w = cfg.height, cfg.width
    gt_labels = np.zeros((h, w), dtype=np.int32)

    # 1. Build layers
    layer_pairs = _build_layer_y(
        h=h,
        n_layers=cfg.n_layers,
        dip_deg=cfg.dip_deg,
        thickness_jitter=cfg.layer_thickness_jitter,
        warp_amplitude=cfg.warp_amplitude,
        rng=rng,
    )
    field = _render_scalar_field(
        h=h,
        w=w,
        layer_pairs=layer_pairs,
        warp_amplitude=cfg.warp_amplitude,
        dip_deg=cfg.dip_deg,
        rng=rng,
    )
    # Assign initial layer labels: layer i (top to bottom) gets label i+1
    for i, _ in enumerate(layer_pairs):
        mask = (field > 0) & (
            np.abs(field - _layer_value(i, cfg.n_layers)) < 1e-3
        )
        gt_labels[mask] = i + 1

    components: list[dict[str, Any]] = [
        {
            "kind": "layers",
            "params": {
                "n_layers": cfg.n_layers,
                "dip_deg": cfg.dip_deg,
                "thickness_jitter": cfg.layer_thickness_jitter,
                "warp_amplitude": cfg.warp_amplitude,
            },
        }
    ]

    # 2. Optional fault
    if cfg.has_fault:
        _apply_fault(
            field=field,
            gt_labels=gt_labels,
            fault_x_frac=cfg.fault_x_frac,
            offset=cfg.fault_offset,
        )
        components.append({"kind": "fault", "params": {"x_frac": cfg.fault_x_frac, "offset": cfg.fault_offset}})

    # 3. Optional channel
    if cfg.has_channel:
        _apply_channel(
            field=field,
            gt_labels=gt_labels,
            y_centre_frac=cfg.channel_y_frac,
            amplitude=cfg.channel_amplitude,
            thickness=cfg.channel_thickness,
            rng=rng,
        )
        components.append(
            {
                "kind": "channel",
                "params": {
                    "y_frac": cfg.channel_y_frac,
                    "amplitude": cfg.channel_amplitude,
                    "thickness": cfg.channel_thickness,
                },
            }
        )

    # 4. Optional unconformity
    if cfg.has_unconformity:
        _apply_unconformity(
            field=field,
            gt_labels=gt_labels,
            y_frac=cfg.unconformity_y_frac,
            dip_below=cfg.unconformity_dip_below,
            dip_above=cfg.unconformity_dip_above,
            rng=rng,
        )
        components.append(
            {
                "kind": "unconformity",
                "params": {
                    "y_frac": cfg.unconformity_y_frac,
                    "dip_below": cfg.unconformity_dip_below,
                    "dip_above": cfg.unconformity_dip_above,
                },
            }
        )

    # 5. Render
    panel_rgb = map_array(field, cfg.colormap)

    # Build label -> scalar value mapping (so evaluate.py can recover)
    label_to_value: dict[int, float] = {}
    for lbl in sorted(set(gt_labels.flatten().tolist())):
        if lbl == 0:
            continue
        v = float(field[gt_labels == lbl].mean())
        label_to_value[int(lbl)] = v

    meta = {
        "n_layers": cfg.n_layers,
        "colormap": cfg.colormap,
        "height": h,
        "width": w,
        "components": components,
        "label_to_value": label_to_value,
        "seed": cfg.seed,
    }
    return panel_rgb, gt_labels, meta
