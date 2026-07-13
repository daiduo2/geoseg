"""Demo for session_state module.

Run: python -m geoseg.session_state_demo
"""

from pathlib import Path

from geoseg.session_state import (
    BacktrackStage,
    ClassificationRecord,
    ExportRecord,
    FigureStatus,
    PanelSelection,
    SegmentationAttempt,
    SegmentationRecord,
    create_session,
    get_summary,
    backtrack,
    load_session,
    save_session,
    update_figure,
)


def main() -> None:
    output_path = Path("runs/session_demo/state.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create session with 3 figures
    print("=== 1. Create session ===")
    state = create_session([
        "runs/M0.5/fig1.png",
        "runs/M0.5/fig2.png",
        "runs/M0.5/fig3.png",
    ])
    print(f"Session {state.session_id}: {len(state.workset)} figures")

    # 2. Classify fig1
    print("\n=== 2. Classify fig1 ===")
    state = update_figure(
        state,
        "fig1",
        status=FigureStatus.CLASSIFIED,
        classification=ClassificationRecord(
            figure_type="velocity_model",
            confidence=0.92,
            reason="Colored layers with velocity colorbar",
        ),
    )
    print(f"fig1 status: {state.workset[0].status.value}")

    # 3. Detect panels for fig1
    print("\n=== 3. Detect panels ===")
    state = update_figure(
        state,
        "fig1",
        status=FigureStatus.CLASSIFIED,
        panels=PanelSelection(
            detected=[
                {"id": 0, "bbox": [0, 0, 600, 400], "source": "cv_detect"},
                {"id": 1, "bbox": [600, 0, 600, 400], "source": "cv_detect"},
            ],
            target_panel_id=1,
        ),
    )
    print(f"fig1 target panel: {state.workset[0].panels.target_panel_id if state.workset[0].panels else None}")

    # 4. Segment fig1
    print("\n=== 4. Segment fig1 ===")
    state = update_figure(
        state,
        "fig1",
        status=FigureStatus.SEGMENTED,
        segmentation=SegmentationRecord(
            result_dir="runs/sandbox/fig1/",
            engine="kmeans_full",
            n_layers=5,
            quality_score=0.85,
            overlay_path="runs/sandbox/fig1/overlay.png",
            labels_path="runs/sandbox/fig1/labels.npy",
            attempts=[
                SegmentationAttempt(engine="v4_kmeans", n_layers=5, quality_score=0.72, notes="Initial try"),
                SegmentationAttempt(engine="kmeans_full", n_layers=5, quality_score=0.85, notes="Best result"),
            ],
        ),
    )
    print(f"fig1 segmented: {state.workset[0].segmentation.engine if state.workset[0].segmentation else None}")

    # 5. Skip fig2
    print("\n=== 5. Skip fig2 ===")
    state = update_figure(
        state,
        "fig2",
        status=FigureStatus.SKIPPED,
        skip_reason="Not a velocity model (shot gather)",
    )
    print(f"fig2 status: {state.workset[1].status.value}")

    # 6. Get summary
    print("\n=== 6. Session summary ===")
    summary = get_summary(state)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 7. Save session
    print(f"\n=== 7. Save to {output_path} ===")
    save_session(state, output_path)
    print("Saved.")

    # 8. Load session
    print(f"\n=== 8. Load from {output_path} ===")
    loaded = load_session(output_path)
    print(f"Loaded session {loaded.session_id} with {len(loaded.workset)} figures")

    # 9. Backtrack fig1 to panel stage
    print("\n=== 9. Backtrack fig1 to 'panel' ===")
    state = backtrack(state, "fig1", BacktrackStage.PANEL)
    entry = state.workset[0]
    print(f"fig1 status: {entry.status.value}")
    print(f"fig1 classification preserved: {entry.classification.figure_type if entry.classification else None}")
    print(f"fig1 panels cleared: {entry.panels is None}")
    print(f"fig1 segmentation cleared: {entry.segmentation is None}")

    # 10. Backtrack fig1 to classify stage
    print("\n=== 10. Backtrack fig1 to 'classify' ===")
    state = backtrack(state, "fig1", BacktrackStage.CLASSIFY)
    entry = state.workset[0]
    print(f"fig1 status: {entry.status.value}")
    print(f"fig1 classification cleared: {entry.classification is None}")

    # 11. Export fig1
    print("\n=== 11. Export fig1 ===")
    state = update_figure(
        state,
        "fig1",
        status=FigureStatus.EXPORTED,
        export=ExportRecord(
            tomo_xyz="runs/M4/fig1_tomo.xyz",
            parfile_snippet="runs/M4/fig1_Par_file_snippet.txt",
        ),
    )
    print(f"fig1 exported: {state.workset[0].export.tomo_xyz if state.workset[0].export else None}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
