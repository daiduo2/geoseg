export const meta = {
  name: "sandbox-segment",
  description:
    "Agent-native autonomous segmentation of a geophysics panel image into velocity layers. Segments the original panel directly, then lets the audit agent identify text/annotation labels by color and merge them into the background. All engine decisions and visual judgments happen inside agent() calls.",
  phases: [
    { title: "Pre-flight", detail: "Agent reads panel and queries strategy memory" },
    { title: "Text Removal", detail: "Optional: remove annotation text using 3D-schematic-inspired conservative mask + Telea r=3" },
    { title: "Segment", detail: "Agent chooses and runs an engine (mask-aware if text removed)" },
    { title: "Audit", detail: "Agent reads overlay-with-legend and writes RegionalAudit" },
    { title: "Repair", detail: "Executor agent repairs retry labels and re-audits" },
    { title: "Finalize", detail: "Save meta, update strategy memory, report result" },
  ],
};

const PANEL_PATH = args.panel_path;
const N_LAYERS_HINT = args.n_layers ?? 0;
const OVERSEGMENT = args.oversegment ?? false;
const ENGINE_HINT = args.engine ?? "";
const PANEL_ID = PANEL_PATH.split("/").pop().replace(/\.[^.]+$/, "");
const OUTPUT_DIR = args.output_dir ?? `runs/sandbox/${PANEL_ID}`;
const MAX_ITERATIONS = args.max_iterations ?? 3;
const TEXT_REMOVE = args.text_remove ?? false;

const AUDIT_SCHEMA = {
  type: "object",
  properties: {
    frozen_labels: { type: "array", items: { type: "integer" } },
    retry_labels: { type: "array", items: { type: "integer" } },
    text_labels: { type: "array", items: { type: "integer" } },
    notes: { type: "string" },
    repair_strategy: {
      type: "string",
      enum: ["regional_fusion", "merge_labels", "remove_text_labels", "switch_engine", "post_process", "accept"],
    },
    secondary_engine: { type: "string" },
    local_fixes: { type: "array", items: { type: "object" } },
    iteration: { type: "integer" },
  },
  required: ["frozen_labels", "retry_labels", "repair_strategy", "iteration"],
};

function phasePreFlight() {
  return agent(
    `You are the pre-flight agent for sandbox-segment.\n\n` +
    `Inputs:\n` +
    `- panel_path: ${PANEL_PATH}\n` +
    `- output_dir: ${OUTPUT_DIR}\n` +
    `- n_layers_hint: ${N_LAYERS_HINT || "(agent decides)"}\n` +
    `- text_remove: ${TEXT_REMOVE}\n\n` +
    `Steps:\n` +
    `1. Read the original panel and visually assess saturation, boundaries, noise, and text/annotation colors.\n` +
    `2. Query strategy_memory for historical hints using the original panel:\n` +
    `   uv run python -c "from geoseg.modules.segment_engines.strategy_memory import query_similar, load_templates; import numpy as np; from PIL import Image; img=np.array(Image.open('${PANEL_PATH}').convert('RGB')); print('SIMILAR', query_similar(img, top_k=3)); print('TEMPLATES', load_templates())"\n` +
    `3. Write ${OUTPUT_DIR}/preflight.json with keys: visual_analysis, memory_hints, recommended_strategy, n_layers_estimate, text_remove_plan.\n` +
    `   - If text_remove=true, include a plan for removing text before segmentation, drawing on the 3D schematic experience (Telea r=3, brightness filter, strict area/aspect).\n\n` +
    `Return ONLY the path to preflight.json.`,
    { label: "preflight", phase: "Pre-flight" }
  );
}

function phaseTextRemove() {
  return agent(
    `You are the text-removal agent for sandbox-segment, drawing on the 3D schematic text-removal experience.\n\n` +
    `Inputs:\n` +
    `- panel_path: ${PANEL_PATH}\n` +
    `- output_dir: ${OUTPUT_DIR}\n\n` +
    `Required actions (use Bash inline Python via uv run python -c):\n` +
    `1. Read the original panel and visually locate text/annotations (BM, PM, LV-N, LV-S, sediment, stars, leader lines, etc.).\n` +
    `2. Call geoseg.modules.text_removal.remove_text with CONSERVATIVE parameters inspired by 3D schematic MSER v2:\n` +
    `   - brightness_thresh: 180-220 (filter out bright geological layers, keep only high-contrast text)\n` +
    `   - max_area: 300-800 (reject large geological regions)\n` +
    `   - max_aspect: 10-15 (reject long stratigraphic boundaries)\n` +
    `   - dilate_iter: 1\n` +
    `   - inpaint_radius: 3 (Telea, the global-best repair strategy from 3D schematic)\n` +
    `   If the resulting mask covers >15% of the panel, tune parameters DOWN. The mask must only cover actual annotation text.\n` +
    `3. Save both files:\n` +
    `   - ${OUTPUT_DIR}/panel_cleaned.jpg (text-removed RGB)\n` +
    `   - ${OUTPUT_DIR}/text_mask.png (binary mask, 255=text)\n` +
    `4. Visually inspect the mask. White pixels should match text/annotations ONLY, not large colored geological regions. If it does not, retry with stricter parameters.\n\n` +
    `Return a JSON object: {text_mask_path, cleaned_path, mask_fraction}.`,
    {
      label: "text-remove",
      phase: "Text Removal",
      schema: {
        type: "object",
        properties: {
          text_mask_path: { type: "string" },
          cleaned_path: { type: "string" },
          mask_fraction: { type: "number" },
        },
        required: ["text_mask_path", "cleaned_path", "mask_fraction"],
      },
    }
  );
}

function phaseSegment(preflightPath, textMaskPath) {
  return agent(
    `You are the segment agent for sandbox-segment.\n\n` +
    `Inputs:\n` +
    `- panel_path: ${PANEL_PATH}\n` +
    `- output_dir: ${OUTPUT_DIR}\n` +
    `- preflight_path: ${preflightPath}\n` +
    `- text_mask_path: ${textMaskPath || "(none, segment on original panel)"}\n` +
    `- n_layers_hint: ${N_LAYERS_HINT || "(use preflight estimate)"}\n` +
    `- oversegment: ${OVERSEGMENT}\n` +
    `- engine_hint: ${ENGINE_HINT || "(agent decides)"}\n\n` +
    `Steps:\n` +
    `1. Read ${preflightPath} and the original panel. If ${textMaskPath ? `text_mask_path is provided (${textMaskPath})` : "no text_mask_path"}, decide how to use it.\n` +
    `2. Decide the initial engine and effective n_layers based on visual analysis + memory hints. Allowed initial engines: v4_kmeans, kmeans_full, edge_guided, edge_grow, ensemble, grayscale, slic_kmeans, lab_l_kmeans, seeded_region_grow.\n` +
    `   - regional_fusion, warm_merge, edge_guided_repair, lab_lchannel, etc. are REPAIR strategies, NOT initial engines. Do NOT use them as the primary engine.\n` +
    `   - If engine_hint is set to "${ENGINE_HINT}", use that engine unless there is a clear visual contradiction.\n` +
    `   - For low-contrast / gradient panels where a feature has the same hue as the background (e.g. panel_3's funnel), prefer lab_l_kmeans or seeded_region_grow.\n` +
    `   - If you choose seeded_region_grow, it uses marker-controlled watershed on a color-gradient cost map. You MUST provide seed points in ${OUTPUT_DIR}/seeds.json as [{"y": int, "x": int, "label": int}, ...].\n` +
    `     Seed placement rules:\n` +
    `     a. Get the ACTUAL pixel dimensions first: uv run python -c "from PIL import Image; print(Image.open('${PANEL_PATH}').size)". The coordinates below are for the full-resolution image, not the thumbnail you see.\n` +
    `     b. Read the original panel and place one seed in the visual center of EACH target layer. Do NOT just space seeds evenly by height.\n` +
    `     c. For panel_3 (1740x3480 px), the target layers are: top dark surface/weak zone, blue weak zone, funnel-shaped refractory peridotite residues, orange mantle, yellow mantle base. Example seed set (adjust if needed): [{"y":200,"x":870,"label":1}, {"y":800,"x":870,"label":2}, {"y":1300,"x":870,"label":3}, {"y":2000,"x":870,"label":4}, {"y":3000,"x":870,"label":5}].\n` +
    `     d. Keep seeds away from white/black text annotations and leader lines.\n` +
    `     e. Use label IDs 1..n_layers; they will be compacted later, so gaps are OK.\n` +
    `3. Run the engine on the panel via Bash inline Python (uv run python -c), save:\n` +
    `   - ${OUTPUT_DIR}/labels.npz\n` +
    `   - ${OUTPUT_DIR}/overlay_legend.jpg (use geoseg.modules.segment_engines.regional_fusion.generate_overlay_with_legend with the original panel)\n` +
    `   ${textMaskPath ? `IMPORTANT: Because a text mask exists, use geoseg.modules.segment_engines.mask_aware.segment_with_text_mask(engine_name, original_panel_rgb, text_mask_bool, n_layers). ` +
    `This runs the engine on the original panel while excluding text pixels from clustering, then assigns text pixels to the nearest non-text label. ` +
    `Save the resulting labels and regenerate overlay_legend.jpg on the original panel.` : "Run the engine on the ORIGINAL panel (no text mask)."}\n` +
    `4. If oversegment=true OR the panel has soft gradients / low-contrast layers (like panel_3's funnel on an orange gradient), use OVERSEGMENT-then-merge:\n` +
    `   - Set n_layers = max(target+3, target*2, 8).\n` +
    `   - Prefer color-based engines (v4_kmeans or kmeans_full) because ensemble voting tends to collapse gradient regions back together, defeating oversegment.\n` +
    `   - The audit agent will later merge gradient fragments that belong to the same layer.\n` +
    `5. Write ${OUTPUT_DIR}/segment_meta.json with engine, n_layers, n_layers_target, text_removed (${TEXT_REMOVE}), notes (including whether oversegment was used).\n\n` +
    `Return ONLY the path to segment_meta.json.`,
    { label: "segment", phase: "Segment" }
  );
}

function phaseAudit(segmentMetaPath) {
  return agent(
    `You are the audit agent for sandbox-segment.\n\n` +
    `Inputs:\n` +
    `- panel_path: ${PANEL_PATH}\n` +
    `- output_dir: ${OUTPUT_DIR}\n` +
    `- segment_meta_path: ${segmentMetaPath}\n\n` +
    `Steps:\n` +
    `1. Read ${OUTPUT_DIR}/overlay_legend.jpg (primary), ${PANEL_PATH}, and any auxiliary views in ${OUTPUT_DIR}/visual_audit/views/ if present.\n` +
    `2. Generate audit materials if missing:\n` +
    `   uv run python -c "import sys; sys.path.insert(0,'src'); import json, pathlib, numpy as np; from PIL import Image; from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend; from geoseg.modules.visual_audit import create_audit_views, save_views, create_audit_crops, save_crops; from geoseg.modules.visual_audit.semantic import compute_semantic_fidelity; labels=np.load('${OUTPUT_DIR}/labels.npz')['labels']; img=np.array(Image.open('${PANEL_PATH}').convert('RGB')); out_dir=pathlib.Path('${OUTPUT_DIR}/visual_audit'); out_dir.mkdir(parents=True, exist_ok=True); overlay=generate_overlay_with_legend(img, labels); Image.fromarray(overlay).save('${OUTPUT_DIR}/overlay_legend.jpg', quality=90); views=create_audit_views(labels, img); save_views(views, str(out_dir/'views')); crops=create_audit_crops(img); save_crops(crops, str(out_dir/'crops')); semantic=compute_semantic_fidelity(labels, img); label_color_map={}; [label_color_map.update({str(int(lbl)): {'area_frac': round(float((labels==lbl).sum()/labels.size),4), 'median_y': round(float(np.median(np.where(labels==lbl)[0])),1) if (labels==lbl).any() else None, 'color': overlay[np.where(labels==lbl)[0][0], np.where(labels==lbl)[1][0]].tolist() if (labels==lbl).any() else [128,128,128]}}) for lbl in sorted(set(labels.flatten())-{0})]; (out_dir/'report.json').write_text(json.dumps({'diagnostic_signals': semantic, 'label_color_map': label_color_map}, indent=2), encoding='utf-8'); print('audit materials ready')"\n` +
    `3. Decide frozen/retry/text labels and a repair strategy.\n` +
    `   - Compare the overlay to the original panel. Any label whose dominant color matches text/annotation ink (white, black, bright yellow) AND occupies a small area (typically <1% of the panel) or forms thin strokes on top of a layer should go into text_labels.\n` +
    `   - NEVER remove a large solid-color layer just because it contains a text annotation; only remove the text label itself.\n` +
    `   - Check for OVER-SEGMENTED / DISCONNECTED FRAGMENTS: compute connected components of each label. If a label contains disconnected islands that are clearly part of the same geological layer (same color, same vertical position, surrounded by the same host label), add them to local_fixes with action "merge_labels" and set repair_strategy to "merge_labels".\n` +
    `     Rules for merging fragments:\n` +
    `     * ONLY merge fragments that are visually the same layer (similar color / same stratigraphic position).\n` +
    `     * NEVER merge labels with clearly different colors or different vertical positions into one layer.\n` +
    `     * For tiny specks (area < ~0.1% of panel) that are not a real layer, use action "remove_text_label" instead.\n` +
    `   - If segment_meta says oversegment was used and you see many small gradient fragments, use repair_strategy: "merge_labels" and list which labels belong to the same layer.\n` +
    `   - If the only issue is text labels, use repair_strategy: "remove_text_labels".\n` +
    `   - If you are unsure whether a label is text or geology, leave it in retry_labels with repair_strategy "regional_fusion" or "post_process" rather than removing it.\n` +
    `   - Before using repair_strategy: "accept", verify there are no clearly spurious small fragments that should be merged.\n` +
    `4. Write ${OUTPUT_DIR}/regional_audit.json matching the required schema.\n\n` +
    `Rules:\n` +
    `- Reference regions by label ID (with color/position as backup).\n` +
    `- Do NOT output PASS/FAIL. Output findings and repair directions only.\n` +
    `- If retry_labels is empty and no text_labels need removal, use repair_strategy: "accept".\n\n` +
    `Return ONLY the parsed RegionalAudit JSON object.`,
    { label: "audit", phase: "Audit", schema: AUDIT_SCHEMA }
  );
}

function phaseRepair(audit, iteration) {
  return agent(
    `You are the repair executor for sandbox-segment, iteration ${iteration}.\n\n` +
    `Inputs:\n` +
    `- panel_path: ${PANEL_PATH}\n` +
    `- output_dir: ${OUTPUT_DIR}\n` +
    `- audit: ${JSON.stringify(audit)}\n\n` +
    `Steps:\n` +
    `1. Read ${OUTPUT_DIR}/regional_audit.json to confirm current state.\n` +
    `2. Execute the repair_strategy from the audit on the original panel:\n` +
    `   - regional_fusion: run geoseg.modules.segment_engines.regional_fusion.regional_segment with RegionalAudit(frozen_labels, retry_labels, notes, iteration, repair_strategy, secondary_engine, local_fixes), using ${PANEL_PATH} as panel_rgb, save labels.npz and overlay_legend.jpg.\n` +
    `   - merge_labels: for each local_fix, apply the correct merge:\n` +
    `       * merge_labels: geoseg.modules.post_process.merge.merge_labels_by_ids(labels, fix.label_ids, target_id=fix.get('target_id', fix.label_ids[0]))\n` +
    `       * remove_text_label: geoseg.modules.post_process.merge.remove_labels_by_ids(labels, fix.label_ids, fill='nearest')\n` +
    `     Save labels.npz, then regenerate overlay_legend.jpg on the original panel.\n` +
    `     Example:\n` +
    `     uv run python -c "import sys; sys.path.insert(0,'src'); import json, numpy as np; from PIL import Image; from geoseg.modules.post_process.merge import merge_labels_by_ids, remove_labels_by_ids; from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend; labels=np.load('${OUTPUT_DIR}/labels.npz')['labels']; img=np.array(Image.open('${PANEL_PATH}').convert('RGB')); audit=json.load(open('${OUTPUT_DIR}/regional_audit.json')); [merge_labels_by_ids(labels, f['label_ids'], target_id=f.get('target_id', f['label_ids'][0])) for f in audit.get('local_fixes',[]) if f.get('action')=='merge_labels']; [remove_labels_by_ids(labels, f['label_ids'], fill='nearest') for f in audit.get('local_fixes',[]) if f.get('action')=='remove_text_label']; np.savez_compressed('${OUTPUT_DIR}/labels.npz', labels=labels); overlay=generate_overlay_with_legend(img, labels); Image.fromarray(overlay).save('${OUTPUT_DIR}/overlay_legend.jpg', quality=90); print('merge repair done')"\n` +
    `   - remove_text_labels: apply geoseg.modules.post_process.merge.remove_labels_by_ids for audit.text_labels (or local_fixes action remove_text_label), save labels.npz, then regenerate overlay_legend.jpg on the original panel. Vacated pixels will fill from the nearest remaining label, restoring the underlying geology.\n` +
    `   - switch_engine: re-run a different engine on the original panel, save labels.npz and overlay_legend.jpg.\n` +
    `   - post_process: apply geoseg.modules.post_process.merge.filter_small_components(min_area_ratio=0.001) OR horizon_refinement, save labels.npz, then regenerate overlay_legend.jpg.\n` +
    `     Example:\n` +
    `     uv run python -c "import sys; sys.path.insert(0,'src'); import numpy as np; from PIL import Image; from geoseg.modules.post_process.merge import filter_small_components; from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend; labels=np.load('${OUTPUT_DIR}/labels.npz')['labels']; img=np.array(Image.open('${PANEL_PATH}').convert('RGB')); labels=filter_small_components(labels, min_area_ratio=0.001); np.savez_compressed('${OUTPUT_DIR}/labels.npz', labels=labels); overlay=generate_overlay_with_legend(img, labels); Image.fromarray(overlay).save('${OUTPUT_DIR}/overlay_legend.jpg', quality=90); print('post_process done')"\n` +
    `3. Increment iteration in regional_audit.json (or the next audit agent will set it).\n\n` +
    `Return a short confirmation string.`,
    { label: `repair-${iteration}`, phase: "Repair" }
  );
}

function phaseFinalize(iterations) {
  return agent(
    `You are the finalize agent for sandbox-segment.\n\n` +
    `Inputs:\n` +
    `- panel_path: ${PANEL_PATH}\n` +
    `- output_dir: ${OUTPUT_DIR}\n` +
    `- iterations: ${iterations}\n\n` +
    `Steps:\n` +
    `1. Read final labels.npz and overlay_legend.jpg.\n` +
    `2. Write ${OUTPUT_DIR}/meta.json with panel_id, panel_path, engine, n_layers, n_layers_target, refinement_applied, output_files.\n` +
    `3. Update strategy_memory by calling record_attempt.\n` +
    `4. Write ${OUTPUT_DIR}/strategy.log documenting engines tried, repairs applied, remaining issues.\n\n` +
    `Return ONLY the path to meta.json.`,
    { label: "finalize", phase: "Finalize" }
  );
}

phase("Pre-flight");
const preflightPath = await phasePreFlight();

let textMaskPath = null;
let cleanedPath = null;
if (TEXT_REMOVE) {
  phase("Text Removal");
  const textRemoveResult = await phaseTextRemove();
  textMaskPath = textRemoveResult?.text_mask_path ?? null;
  cleanedPath = textRemoveResult?.cleaned_path ?? null;
  if (!textMaskPath) {
    throw new Error("Text removal was requested but did not return a valid text_mask_path");
  }
}

phase("Segment");
const segmentMetaPath = await phaseSegment(preflightPath, textMaskPath);

phase("Audit");
let audit = await phaseAudit(segmentMetaPath);

phase("Repair");
let iteration = audit.iteration ?? 1;
while (iteration <= MAX_ITERATIONS) {
  if (audit.repair_strategy === "accept" || (audit.retry_labels.length === 0 && (audit.text_labels ?? []).length === 0)) {
    break;
  }
  await phaseRepair(audit, iteration);
  audit = await phaseAudit(segmentMetaPath);
  iteration += 1;
}

phase("Finalize");
const metaPath = await phaseFinalize(iteration);

return {
  output_dir: OUTPUT_DIR,
  meta_path: metaPath,
  labels_path: `${OUTPUT_DIR}/labels.npz`,
  overlay_path: `${OUTPUT_DIR}/overlay_legend.jpg`,
  iterations: iteration,
};
