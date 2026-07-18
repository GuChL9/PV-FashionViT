import pytest

from src.analyze_clipping_effect import summarize_clipping_effect


def test_summarize_clipping_effect_splits_inner_grid_and_outer_ring():
    rows = [
        {"model": "demo", "dx": 0, "dy": 0, "accuracy": 0.9},
        {"model": "demo", "dx": 12, "dy": -12, "accuracy": 0.7},
        {"model": "demo", "dx": 18, "dy": 0, "accuracy": 0.5},
        {"model": "demo", "dx": -18, "dy": 18, "accuracy": 0.3},
    ]

    summary = summarize_clipping_effect(rows, no_clip_limit=14)

    assert len(summary) == 1
    assert summary[0]["inner_points"] == 2
    assert summary[0]["outer_points"] == 2
    assert summary[0]["inner_accuracy"] == pytest.approx(0.8)
    assert summary[0]["outer_accuracy"] == pytest.approx(0.4)
    assert summary[0]["edge_gap"] == pytest.approx(0.4)


def test_summarize_clipping_effect_requires_both_regions():
    rows = [{"model": "demo", "dx": 0, "dy": 0, "accuracy": 0.9}]

    with pytest.raises(ValueError, match="both inner and outer"):
        summarize_clipping_effect(rows)
