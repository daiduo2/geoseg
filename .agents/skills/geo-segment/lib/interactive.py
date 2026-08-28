"""Human-in-the-loop correction utilities.

Review.json schema (v2):

    {
      "panel_id": int,
      "path": "jet_vivid" | "pastel_faded",
      "zones": [
        {
          "color_name": "red",
          "vlm_xy": [vx, vy],          // original VLM rep point
          "internal_xy": [ix, iy],     // algorithm-derived internal point
          "current_xy": [cx, cy],      // point currently used (== internal)
          "needs_fix": false,
          "corrected_xy": null,        // override: absolute new point
          "nudge_xy": [0, 0],          // override: relative shift from current_xy
          "comment": ""
        }
      ]
    }

``corrected_xy`` takes precedence over ``nudge_xy``.  Either one triggers
re-segmentation with updated seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

from .segment import SegmentResult, segment_jet_vivid


def write_review_prompts(out_dir: str | Path, result: SegmentResult, panel_idx: int) -> Path:
    """Emit ``review.json`` so the user can mark zones that need re-seeding.

    Each zone exposes three coordinate layers:
    - ``vlm_xy``:  where the VLM originally said the colour was
    - ``internal_xy``: where the algorithm eroded to
    - ``current_xy``:  the seed actually used (== internal_xy until user edits)

    If a zone's seed landed on background, ``comment`` is auto-populated with a
    warning so the user knows to double-check.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes = result.notes.get("reps_refined", [])
    zones = []
    for idx, name in enumerate(result.color_names):
        entry = {
            "color_name": name,
            "needs_fix": False,
            "corrected_xy": None,
            "nudge_xy": [0, 0],
            "comment": "",
        }
        if notes and idx < len(notes):
            n = notes[idx]
            entry["vlm_xy"] = [n.get("vlm_x", 0), n.get("vlm_y", 0)]
            entry["internal_xy"] = [
                n.get("internal_x") or n.get("vlm_x") or 0,
                n.get("internal_y") or n.get("vlm_y") or 0,
            ]
            entry["current_xy"] = entry["internal_xy"].copy()
            if n.get("on_background"):
                entry["comment"] = (
                    "WARNING: algorithm seed appears to be on background / grey area. "
                    "Please verify in panel_{panel_id}_segmentation.png. "
                    "If wrong, set needs_fix=true and provide corrected_xy."
                )
        else:
            entry["vlm_xy"] = [0, 0]
            entry["internal_xy"] = [0, 0]
            entry["current_xy"] = [0, 0]
        zones.append(entry)

    suspicious = sum(1 for z in zones if "WARNING" in z.get("comment", ""))
    review = {
        "panel_id": panel_idx,
        "path": result.path,
        "suspicious_zones": suspicious,
        "zones": zones,
        "instructions": (
            "Review each zone against panel_{panel_id}_segmentation.png.\n"
            "  vlm_xy      = where the VLM originally pointed\n"
            "  internal_xy = where the algorithm eroded to\n"
            "  current_xy  = seed actually used (starts == internal_xy)\n\n"
            "To fix a zone, set needs_fix=true and EITHER:\n"
            "  corrected_xy=[x, y]   (absolute panel-local pixel coords)\n"
            "  nudge_xy=[dx, dy]     (relative shift from current_xy)\n\n"
            "Then run: geo-segment --apply-review review.json"
        ),
    }
    path = out / "review.json"
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_corrections(
    panel_rgb,
    review: dict,
) -> SegmentResult:
    """Build new representative points from review JSON and re-segment.

    Only ``jet_vivid`` path supports correction (pastel seeds are fixed by
    the colorbar).
    """
    reps = []
    for z in review["zones"]:
        xy = None
        if not z.get("needs_fix"):
            xy = z.get("current_xy")
        else:
            if z.get("corrected_xy") is not None:
                xy = z["corrected_xy"]
            elif z.get("current_xy") is not None:
                cx, cy = z["current_xy"]
                dx, dy = z.get("nudge_xy", [0, 0])
                xy = [cx + dx, cy + dy]
        if xy is None:
            raise ValueError(
                f"Zone '{z.get('color_name', '?')}' has no usable coordinates. "
                "Set current_xy or corrected_xy."
            )
        reps.append({
            "color_name": z["color_name"],
            "representative_point": {"x": int(xy[0]), "y": int(xy[1])},
        })
    if not reps:
        raise ValueError("review.json has no usable points")
    return segment_jet_vivid(panel_rgb, reps)


__all__ = ["write_review_prompts", "apply_corrections"]
