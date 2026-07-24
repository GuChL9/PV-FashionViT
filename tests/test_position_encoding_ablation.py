import json

import pytest

from src.analyze_position_encoding_ablation import PROFILES, collect_rows, summarize_rows


def _payload(value):
    return {
        "summary": {
            "center_accuracy": value,
            "large_shift_accuracy": value - 0.1,
            "rotation_accuracy": value - 0.2,
            "shift_rotation_accuracy": value - 0.3,
            "robust_drop": 0.1,
        }
    }


def test_position_encoding_collection_and_summary(tmp_path):
    seeds = [42, 2026, 3407]
    for profile_index, profile in enumerate(PROFILES):
        for seed_index, seed in enumerate(seeds):
            run_dir = tmp_path / f"{profile.run_base}_s{seed}"
            run_dir.mkdir(parents=True)
            value = 0.6 + profile_index * 0.05 + seed_index * 0.01
            (run_dir / "evaluation.json").write_text(
                json.dumps(_payload(value)), encoding="utf-8"
            )

    rows = collect_rows(seeds, tmp_path)
    summary = summarize_rows(rows, seeds)

    assert len(rows) == len(PROFILES) * len(seeds)
    assert len(summary) == len(PROFILES)
    assert summary[0]["center_accuracy_mean"] == pytest.approx(0.61)
    assert summary[2]["large_shift_accuracy_mean"] == pytest.approx(0.61)
