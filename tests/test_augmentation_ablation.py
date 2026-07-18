import json

import pytest

from src.analyze_augmentation_ablation import STAGES, collect_rows, summarize_rows


def _summary(center, large, rotation, combined):
    return {
        "summary": {
            "center_accuracy": center,
            "large_shift_accuracy": large,
            "rotation_accuracy": rotation,
            "shift_rotation_accuracy": combined,
            "robust_drop": center - large,
        }
    }


def test_collect_rows_uses_run_evaluations_and_global_endpoints(tmp_path):
    outputs = tmp_path / "outputs"
    tables = outputs / "tables"
    tables.mkdir(parents=True)
    global_table = tables / "main_results.csv"
    global_table.write_text(
        "model,center_accuracy,large_shift_accuracy,rotation_accuracy,shift_rotation_accuracy,robust_drop\n"
        "vit_abspos_center_cpu,0.86,0.20,0.46,0.17,0.66\n"
        "vit_aug_cpu,0.63,0.65,0.65,0.65,-0.02\n",
        encoding="utf-8",
    )
    stage_values = {
        "vit_shift_cpu_s42": _summary(0.70, 0.68, 0.40, 0.39),
        "vit_shift_rotation_cpu_s42": _summary(0.64, 0.66, 0.64, 0.65),
    }
    for run_name, payload in stage_values.items():
        run_dir = outputs / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = collect_rows(42, outputs, global_table)

    assert [row["stage"] for row in rows] == [stage.key for stage in STAGES]
    assert rows[1]["large_shift_accuracy"] == 0.68
    assert rows[2]["rotation_accuracy"] == 0.64
    assert rows[3]["robust_drop"] == -0.02


def test_summarize_rows_reports_sample_standard_deviation():
    rows = []
    for seed, offset in [(42, 0.0), (2026, 0.1), (3407, 0.2)]:
        for stage in STAGES:
            rows.append(
                {
                    "stage": stage.key,
                    "label": stage.label,
                    "short_label": stage.short_label,
                    "seed": seed,
                    "center_accuracy": 0.5 + offset,
                    "large_shift_accuracy": 0.4 + offset,
                    "rotation_accuracy": 0.3 + offset,
                    "shift_rotation_accuracy": 0.2 + offset,
                    "robust_drop": 0.1,
                }
            )

    summary = summarize_rows(rows, [42, 2026, 3407])

    assert len(summary) == len(STAGES)
    assert summary[0]["center_accuracy_mean"] == pytest.approx(0.6)
    assert summary[0]["center_accuracy_std"] == pytest.approx(0.1)
