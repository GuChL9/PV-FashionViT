"""Create report figures for the E3-E6 control experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("vit_abspos_center_cpu", "ViT-AbsPos\ncenter"),
    ("vit_shiftrot_noerase_cpu", "ViT\nshift+rot"),
    ("vit_aug_cpu", "ViT-Aug"),
    ("vit_meanpool_center_cpu", "ViT-Mean\ncenter"),
    ("vit_meanpool_cpu", "ViT-Mean"),
    ("hybrid_vit_cls_cpu", "Hybrid\nCLS"),
    ("hybrid_vit_cpu", "Hybrid\nMean"),
]

COMPARISONS = [
    ("Shift+Rotation", "vit_abspos_center_cpu", "vit_shiftrot_noerase_cpu"),
    ("Random\nErasing", "vit_shiftrot_noerase_cpu", "vit_aug_cpu"),
    ("Mean Pool\n(center)", "vit_abspos_center_cpu", "vit_meanpool_center_cpu"),
    ("Mean Pool\n(aug)", "vit_aug_cpu", "vit_meanpool_cpu"),
    ("Conv Stem\n(CLS fixed)", "vit_aug_cpu", "hybrid_vit_cls_cpu"),
    ("Mean Pool\n(Hybrid)", "hybrid_vit_cls_cpu", "hybrid_vit_cpu"),
]

METRICS = [
    ("center_accuracy", "Center"),
    ("large_shift_accuracy", "Large Shift"),
    ("rotation_accuracy", "Rotation"),
    ("grid_accuracy", "Grid"),
]

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_records(root: Path) -> dict[str, dict]:
    records = {}
    for run_name, display_name in RUNS:
        run_dir = root / run_name
        evaluation_path = run_dir / "evaluation.json"
        grid_path = run_dir / "tables" / "grid_accuracy.csv"
        per_class_path = run_dir / "tables" / "per_class_accuracy.csv"
        if not evaluation_path.exists():
            raise FileNotFoundError(f"Missing evaluation file: {evaluation_path}")
        if not grid_path.exists():
            raise FileNotFoundError(f"Missing grid file: {grid_path}")
        if not per_class_path.exists():
            raise FileNotFoundError(f"Missing per-class file: {per_class_path}")
        with evaluation_path.open(encoding="utf-8") as stream:
            evaluation = json.load(stream)
        grid = pd.read_csv(grid_path)
        if len(grid) != 49:
            raise ValueError(f"{grid_path} should contain 49 rows, found {len(grid)}")
        summary = dict(evaluation["summary"])
        summary["grid_accuracy"] = float(grid["accuracy"].mean())
        records[run_name] = {
            "run_name": run_name,
            "display_name": display_name,
            "summary": summary,
            "grid": grid,
            "per_class": pd.read_csv(per_class_path),
        }
    return records


def annotation_color(cmap_name: str, value: float, vmin: float, vmax: float) -> str:
    rgba = plt.get_cmap(cmap_name)(plt.Normalize(vmin=vmin, vmax=vmax)(value))
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    return "black" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "white"


def plot_metrics_comparison(records: dict[str, dict], output: Path) -> None:
    labels = [records[name]["display_name"] for name, _ in RUNS]
    x = np.arange(len(labels))
    width = 0.19
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    figure, axis = plt.subplots(figsize=(13.5, 6.2))
    for index, (metric, label) in enumerate(METRICS):
        values = [records[name]["summary"][metric] * 100 for name, _ in RUNS]
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=label, color=colors[index])
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}",
                      ha="center", va="bottom", fontsize=7)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Control experiments: accuracy across evaluation protocols", pad=18)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    figure.subplots_adjust(bottom=0.2, top=0.88)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def plot_factor_deltas(records: dict[str, dict], output: Path) -> None:
    labels, grid_delta, large_delta = [], [], []
    for label, before, after in COMPARISONS:
        labels.append(label)
        grid_delta.append((records[after]["summary"]["grid_accuracy"] - records[before]["summary"]["grid_accuracy"]) * 100)
        large_delta.append((records[after]["summary"]["large_shift_accuracy"] - records[before]["summary"]["large_shift_accuracy"]) * 100)
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    bars_grid = axis.bar(x - width / 2, grid_delta, width, label="Grid Acc delta", color="#3B82B6")
    bars_large = axis.bar(x + width / 2, large_delta, width, label="Large Shift delta", color="#E45756")
    axis.axhline(0, color="#333333", linewidth=1)
    for bars in (bars_grid, bars_large):
        for bar in bars:
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            offset = 0.8 if value >= 0 else -0.8
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}",
                      ha="center", va=va, fontsize=8)
    axis.set_ylabel("Change (percentage points)")
    axis.set_title("Single-variable control deltas")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def plot_grid_heatmaps(records: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.4, 7.7), layout="constrained")
    image = None
    for axis in axes.flat[len(RUNS):]:
        axis.set_axis_off()
    for axis, (run_name, _) in zip(axes.flat, RUNS):
        record = records[run_name]
        grid = record["grid"]
        values = sorted(grid["dx"].unique())
        lookup = {(int(row.dx), int(row.dy)): float(row.accuracy) for row in grid.itertuples()}
        matrix = np.array([[lookup[(dx, dy)] for dx in values] for dy in values])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
        axis.set_title(record["display_name"], fontsize=11)
        axis.set_xticks(range(len(values)), values, fontsize=8)
        axis.set_yticks(range(len(values)), values, fontsize=8)
        axis.set_xlabel("dx")
        axis.set_ylabel("dy")
        for iy, row in enumerate(matrix):
            for ix, value in enumerate(row):
                axis.text(ix, iy, f"{value:.2f}", ha="center", va="center",
                          fontsize=6.2, color=annotation_color("YlGnBu", value, 0, 1))
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, label="Accuracy")
    figure.suptitle("49-position grid accuracy for main and control runs", fontsize=15)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def plot_per_class(records: dict[str, dict], output: Path) -> None:
    selected = ["vit_aug_cpu", "vit_meanpool_cpu", "hybrid_vit_cls_cpu", "hybrid_vit_cpu"]
    labels = [records[name]["display_name"].replace("\n", " ") for name in selected]
    matrix = []
    for name in selected:
        table = records[name]["per_class"]
        matrix.append([float(table.loc[table["class"] == class_name, "accuracy"].iloc[0]) * 100 for class_name in CLASS_NAMES])
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(12.2, 4.7))
    image = axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    axis.set_title("Large-shift per-class accuracy")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for iy, row in enumerate(matrix):
        for ix, value in enumerate(row):
            axis.text(ix, iy, f"{value:.0f}", ha="center", va="center", fontsize=7,
                      color=annotation_color("YlGnBu", value, 0, 100))
    figure.colorbar(image, ax=axis, label="Accuracy (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=240)
    plt.close(figure)


def copy_to_report(output_dir: Path, report_dir: Path, names: list[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(output_dir / name, report_dir / name)


def main() -> None:
    root = Path("outputs")
    figures_dir = root / "figures"
    report_figures = Path("report") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(root)
    outputs = [
        "control_metrics_comparison.png",
        "control_factor_deltas.png",
        "control_grid_heatmaps.png",
        "control_per_class_large_shift.png",
    ]
    plot_metrics_comparison(records, figures_dir / outputs[0])
    plot_factor_deltas(records, figures_dir / outputs[1])
    plot_grid_heatmaps(records, figures_dir / outputs[2])
    plot_per_class(records, figures_dir / outputs[3])
    copy_to_report(figures_dir, report_figures, outputs)
    print("Generated control experiment figures:")
    for name in outputs:
        print(f"- {figures_dir / name}")
        print(f"- {report_figures / name}")


if __name__ == "__main__":
    main()
