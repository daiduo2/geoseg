from __future__ import annotations

import json

import numpy as np

from geoseg.modules.segment_engines.strategy_memory import (
    analyze_batch,
    load_templates,
    query_similar,
    record_attempt,
    save_templates,
)


def test_strategy_memory_records_queries_and_analyzes(tmp_path):
    memory_path = tmp_path / "strategy_memory.jsonl"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8] = [220, 20, 20]
    image[8:] = [20, 20, 220]

    for idx in range(5):
        record_attempt(
            image,
            engine="edge_guided" if idx < 4 else "v4_kmeans",
            params={"n_layers": 2},
            scores={"boundary_alignment": 0.8 + idx * 0.01},
            outcome="success" if idx < 4 else "retry",
            notes=f"attempt {idx}",
            memory_path=memory_path,
        )

    raw_lines = memory_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 5
    assert json.loads(raw_lines[0])["engine"] == "edge_guided"

    similar = query_similar(image, top_k=2, memory_path=memory_path)
    assert [record["engine"] for record in similar] == ["edge_guided", "edge_guided"]

    analysis = analyze_batch(memory_path=memory_path, min_samples=5)
    assert analysis["engine_success_rates"]["edge_guided"] == 1.0
    assert analysis["summary"]["total_records"] == 5

    template_path = save_templates(analysis, tmp_path / "strategy_templates.json")
    assert load_templates(template_path)["summary"]["total_records"] == 5
