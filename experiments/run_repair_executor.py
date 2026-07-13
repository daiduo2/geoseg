import numpy as np
from PIL import Image
from geoseg.modules.segment_engines.regional_fusion import RegionalAudit, regional_segment, generate_overlay_with_legend
from geoseg.modules.post_process.merge import merge_labels_by_ids, remove_labels_by_ids, filter_small_components
import json
import os

# Paths
panel_path = "/Users/daiduo2/geoseg/experiments/3d_schematic/figures/panels/panel_1.png"
output_dir = "/Users/daiduo2/geoseg/runs/sandbox_workflow_seeded_v3/panel_1"
labels_path = os.path.join(output_dir, "labels.npz")

# Load panel image
panel_rgb = np.array(Image.open(panel_path).convert("RGB"))

# Load existing labels
existing = np.load(labels_path)
labels = existing["labels"]

# Audit data
audit = RegionalAudit(
    frozen_labels=[1],
    retry_labels=[2, 3, 4, 5],
    iteration=1,
    repair_strategy="regional_fusion",
    secondary_engine="",
    local_fixes=[
        {"label": 2, "action": "merge_into_adjacent", "reason": "Tiny 0.27% fragment at crust-plume boundary; artifact from seed bleed at complex boundary"},
        {"label": 4, "action": "merge_with", "target_label": 5, "reason": "Labels 4 and 5 are adjacent fragments of the same orange plume stem (mean_rgb [192,96,59] vs [196,103,67]); they split due to narrow-waist topology but represent one geological body"},
        {"label": 3, "action": "re_segment", "reason": "Label 3 (69% area) severely under-segmented: merges yellow mantle background, blue MLD layer, mantle lithosphere, and most of orange plume head/stem into one region. Only 2 of 5 intended geological layers were separated. Needs re-segmentation with stronger color-distance thresholds or additional seeds"},
    ],
    notes="Seeded region grow with 5 seeds produced only 2 meaningful layers. Continental crust (label 1, 21%) is correct. All other seeds (MLD layer, plume head, plume stem, mantle base) merged into label 3 (69%) because orange plume and yellow mantle have similar hue/saturation. Labels 4 and 5 are small plume-stem fragments that split off due to narrow topology. No text labels were captured. Repair requires re-segmentation of the entire lower portion with adjusted parameters (tighter color threshold, LAB-space distance, or edge-aware watershed) to separate: (a) blue MLD layer/mantle lithosphere, (b) orange plume head+stem, (c) yellow mantle background."
)

# Step 1: Apply local fixes before regional fusion
# Fix label 2: merge into adjacent (merge with label 3 since it's the only adjacent)
labels = merge_labels_by_ids(labels, [2, 3], target_id=3)

# Fix label 4: merge with label 5
labels = merge_labels_by_ids(labels, [4, 5], target_id=5)

# Now labels are: 1 (frozen), 3 (merged 2+3), 5 (merged 4+5)
# But we need to re-segment the retry region. Let's renumber for clarity.
# After merges: 1, 3, 5. We want to re-segment labels 3 and 5 (the retry region).
# Actually, after merging 4+5 into 5, label 5 is the merged plume stem.
# Label 3 is the huge under-segmented region.
# Both 3 and 5 are in retry_labels.

# Save the pre-fusion labels for reference
np.savez(os.path.join(output_dir, "labels_pre_fusion.npz"), labels=labels)

# Step 2: Run regional_segment with the audit
# The regional_segment will freeze label 1 and re-segment the rest with secondary engine
primary_result = {"labels": labels}

result = regional_segment(
    panel_rgb=panel_rgb,
    n_layers=5,
    primary_result=primary_result,
    audit=audit,
)

# Step 3: Save results
fused_labels = result["labels"]
np.savez(os.path.join(output_dir, "labels.npz"), labels=fused_labels)

overlay = result["overlay"]
if overlay is not None:
    Image.fromarray(overlay).save(os.path.join(output_dir, "overlay_legend.jpg"))

# Step 4: Update audit iteration
audit.iteration = 2
audit_dict = {
    "frozen_labels": audit.frozen_labels,
    "retry_labels": audit.retry_labels,
    "iteration": audit.iteration,
    "repair_strategy": audit.repair_strategy,
    "secondary_engine": audit.secondary_engine,
    "local_fixes": audit.local_fixes,
    "notes": audit.notes,
    "meta": result.get("meta", {}),
}
with open(os.path.join(output_dir, "regional_audit.json"), "w") as f:
    json.dump(audit_dict, f, indent=2)

print("Repair executor completed iteration 1.")
print(f"  Saved labels.npz with shape {fused_labels.shape}")
print(f"  Saved overlay_legend.jpg")
print(f"  Saved regional_audit.json with iteration={audit.iteration}")
print(f"  Meta: {result.get('meta', {})}")
