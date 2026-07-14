"""Create report-ready multi-seed tables and figures.

This script reads the 30 formal multi-seed run directories and aggregates
metrics by base model. It does not depend on PyTorch, so it can be run after
training from any Python environment that has matplotlib, numpy, and pandas.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [42, 2026, 3407, 12345, 98765]
MODELS = [
    ("MLP", "mlp_center_cpu"),
    ("CNN", "cnn_center_cpu"),
    ("ViT-AbsPos", "vit_abspos_center_cpu"),
    ("ViT-Aug", "vit_aug_cpu"),
    ("ViT-MeanPool", "vit_meanpool_cpu"),
    ("HybridConv-ViT", "hybrid_vit_cpu"),
]
METRICS = [
    ("center_accuracy", "Center"),
    ("large_shift_accuracy", "Large Shift"),
    ("grid_accuracy", "Grid"),
    ("rotation_accuracy", "Rotation"),
    ("shift_rotation_accuracy", "Shift+Rotation"),
]
COLORS = {
    "MLP": "#4C78A8",
    "CNN": "#F58518",
    "ViT-AbsPos": "#54A24B",
    "ViT-Aug": "#E45756",
    "ViT-MeanPool": "#72B7B2",
    "HybridConv-ViT": "#B279A2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed PV-FashionViT results")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--report-dir", default="report")
    return parser.parse_args()


def read_grid_accuracy(run_dir: Path) -> float:
    grid_path = run_dir / "tables" / "grid_accuracy.csv"
    grid = pd.read_csv(grid_path)
    if len(grid) != 49:
        raise ValueError(f"{grid_path} should contain 49 rows, found {len(grid)}")
    return float(grid["accuracy"].mean())


def read_angle_summary(run_dir: Path) -> tuple[float, float]:
    angle_path = run_dir / "tables" / "angle_accuracy.csv"
    angles = pd.read_csv(angle_path)
    if len(angles) != 7:
        raise ValueError(f"{angle_path} should contain 7 rows, found {len(angles)}")
    values = angles["accuracy"].astype(float)
    return float(values.mean()), float(values.min())


def collect_records(root: Path) -> pd.DataFrame:
    records = []
    missing = []
    for display_name, run_base in MODELS:
        for seed in SEEDS:
            run_name = f"{run_base}_s{seed}"
            run_dir = root / run_name
            evaluation_path = run_dir / "evaluation.json"
            checkpoint_path = run_dir / "checkpoints" / "best.pt"
            if not evaluation_path.exists():
                missing.append(str(evaluation_path))
                continue
            if not checkpoint_path.exists():
                missing.append(str(checkpoint_path))
            with evaluation_path.open(encoding="utf-8") as stream:
                evaluation = json.load(stream)
            summary = evaluation["summary"]
            angle_mean, angle_worst = read_angle_summary(run_dir)
            row = {
                "model": display_name,
                "run_name": run_name,
                "seed": seed,
                "center_accuracy": float(summary["center_accuracy"]),
                "small_shift_accuracy": float(summary["small_shift_accuracy"]),
                "large_shift_accuracy": float(summary["large_shift_accuracy"]),
                "rotation_accuracy": float(summary["rotation_accuracy"]),
                "shift_rotation_accuracy": float(summary["shift_rotation_accuracy"]),
                "grid_accuracy": read_grid_accuracy(run_dir),
                "robust_drop": float(summary["robust_drop"]),
                "rotation_drop": float(summary["rotation_drop"]),
                "fixed_angle_mean": angle_mean,
                "fixed_angle_worst": angle_worst,
            }
            records.append(row)
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))
    table = pd.DataFrame(records)
    if len(table) != len(SEEDS) * len(MODELS):
        raise ValueError(f"Expected 30 records, found {len(table)}")
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "center_accuracy",
        "small_shift_accuracy",
        "large_shift_accuracy",
        "grid_accuracy",
        "rotation_accuracy",
        "shift_rotation_accuracy",
        "robust_drop",
        "rotation_drop",
        "fixed_angle_mean",
        "fixed_angle_worst",
    ]
    rows = []
    for display_name, _ in MODELS:
        subset = table[table["model"] == display_name]
        row = {"model": display_name}
        for metric in metric_columns:
            values = subset[metric].astype(float) * 100
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(table: pd.DataFrame, summary: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "multiseed_raw_results.csv", index=False)
    summary.to_csv(tables_dir / "multiseed_main_results.csv", index=False)

    pairs = [
        ("HybridConv-ViT", "ViT-MeanPool"),
        ("ViT-MeanPool", "ViT-Aug"),
        ("ViT-Aug", "ViT-AbsPos"),
    ]
    rows = []
    for left, right in pairs:
        for seed in SEEDS:
            left_value = table[(table["model"] == left) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            right_value = table[(table["model"] == right) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            rows.append({
                "comparison": f"{left} - {right}",
                "left": left,
                "right": right,
                "seed": seed,
                "grid_delta_pp": (left_value - right_value) * 100,
            })
    pd.DataFrame(rows).to_csv(tables_dir / "multiseed_pairwise_grid_deltas.csv", index=False)


def _style_axis(axis):
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_grid_errorbars(summary: pd.DataFrame, output: Path) -> None:
    labels = summary["model"].tolist()
    means = summary["grid_accuracy_mean"].to_numpy()
    stds = summary["grid_accuracy_std"].to_numpy()
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9.6, 5.2))
    axis.barh(y, means, xerr=stds, color=[COLORS[label] for label in labels], alpha=0.9,
              error_kw={"elinewidth": 1.4, "capsize": 4, "capthick": 1.2})
    for index, (mean, std) in enumerate(zip(means, stds)):
        axis.text(mean + std + 1.0, index, f"{mean:.1f}", va="center", fontsize=9)
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 90)
    axis.set_xlabel("Grid Accuracy (%)")
    axis.set_title("Multi-seed Grid Accuracy")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_metric_errorbars(summary: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(METRICS))
    offsets = np.linspace(-0.30, 0.30, len(MODELS))
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    for offset, (display_name, _) in zip(offsets, MODELS):
        row = summary[summary["model"] == display_name].iloc[0]
        means = np.array([row[f"{metric}_mean"] for metric, _ in METRICS])
        stds = np.array([row[f"{metric}_std"] for metric, _ in METRICS])
        axis.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.2,
            color=COLORS[display_name],
            label=display_name,
        )
    axis.set_xticks(x, [label for _, label in METRICS])
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Multi-seed Accuracy by Evaluation Protocol")
    axis.legend(ncol=3, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_pairwise_deltas(table: pd.DataFrame, output: Path) -> None:
    pairs = [
        ("HybridConv-ViT", "ViT-MeanPool"),
        ("ViT-MeanPool", "ViT-Aug"),
        ("ViT-Aug", "ViT-AbsPos"),
    ]
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    seed_positions = np.arange(len(SEEDS))
    for left, right in pairs:
        deltas = []
        for seed in SEEDS:
            left_value = table[(table["model"] == left) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            right_value = table[(table["model"] == right) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            deltas.append((left_value - right_value) * 100)
        label = f"{left} - {right}"
        axis.plot(seed_positions, deltas, marker="o", linewidth=2, label=label)
    axis.axhline(0, color="#222222", linewidth=1, linestyle="--")
    axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
    axis.set_xlabel("Seed")
    axis.set_ylabel("Grid Accuracy Delta (percentage points)")
    axis.set_title("Paired Grid Accuracy Differences by Seed")
    axis.legend(fontsize=8, frameon=False, loc="upper left")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_angle_summary(summary: pd.DataFrame, output: Path) -> None:
    labels = summary["model"].tolist()
    y = np.arange(len(labels))
    height = 0.36
    figure, axis = plt.subplots(figsize=(9.8, 5.4))
    axis.barh(
        y - height / 2,
        summary["fixed_angle_mean_mean"],
        xerr=summary["fixed_angle_mean_std"],
        height=height,
        color="#4C78A8",
        label="Seven-angle mean",
        error_kw={"elinewidth": 1.3, "capsize": 3},
    )
    axis.barh(
        y + height / 2,
        summary["fixed_angle_worst_mean"],
        xerr=summary["fixed_angle_worst_std"],
        height=height,
        color="#F58518",
        label="Worst scanned angle",
        error_kw={"elinewidth": 1.3, "capsize": 3},
    )
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 90)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title("Fixed-angle Robustness Across Seeds")
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def copy_to_report(figures_dir: Path, report_figures: Path) -> None:
    report_figures.mkdir(parents=True, exist_ok=True)
    for path in figures_dir.glob("multiseed_*.png"):
        shutil.copy2(path, report_figures / path.name)


def main():
    args = parse_args()
    root = Path(args.root)
    report_dir = Path(args.report_dir)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    report_figures = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = collect_records(root)
    summary = summarize(table)
    write_tables(table, summary, tables_dir)
    plot_grid_errorbars(summary, figures_dir / "multiseed_grid_accuracy_errorbars.png")
    plot_metric_errorbars(summary, figures_dir / "multiseed_metric_errorbars.png")
    plot_pairwise_deltas(table, figures_dir / "multiseed_pairwise_grid_deltas.png")
    plot_angle_summary(summary, figures_dir / "multiseed_angle_mean_worst.png")
    copy_to_report(figures_dir, report_figures)
    print("Wrote multi-seed figures:")
    for path in sorted(figures_dir.glob("multiseed_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
