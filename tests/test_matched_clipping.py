import pytest

from src.analyze_matched_clipping import summarize_matched_rows
from src.evaluate_matched_clipping import matched_radius_coordinates


def test_matched_radius_coordinates_control_exact_distance():
    rows = matched_radius_coordinates(max_shift=18, no_clip_limit=14)
    radii = sorted({int(row["radius_squared"]) for row in rows})

    assert radii == [225, 250, 260, 265, 290, 338, 340]
    assert len(rows) == 104
    for radius in radii:
        conditions = {bool(row["clipped"]) for row in rows if row["radius_squared"] == radius}
        assert conditions == {False, True}


def test_summarize_matched_rows_macro_averages_each_radius():
    rows = [
        {"model": "demo", "seed": 42, "radius_squared": 225, "clipped": False, "accuracy": 0.8},
        {"model": "demo", "seed": 42, "radius_squared": 225, "clipped": True, "accuracy": 0.5},
        {"model": "demo", "seed": 42, "radius_squared": 250, "clipped": False, "accuracy": 0.6},
        {"model": "demo", "seed": 42, "radius_squared": 250, "clipped": True, "accuracy": 0.4},
    ]

    radius_rows, summary = summarize_matched_rows(rows)

    assert len(radius_rows) == 2
    assert summary[0]["matched_radii"] == 2
    assert summary[0]["no_clip_accuracy"] == pytest.approx(0.7)
    assert summary[0]["clipped_accuracy"] == pytest.approx(0.45)
    assert summary[0]["matched_gap"] == pytest.approx(0.25)
