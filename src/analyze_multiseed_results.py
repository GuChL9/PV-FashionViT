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
    "MLP": "#244A73",
    "CNN": "#C58B32",
    "ViT-AbsPos": "#4E8B66",
    "ViT-Aug": "#2F7F83",
    "ViT-MeanPool": "#5F6B76",
    "HybridConv-ViT": "#8A5A83",
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


def format_percent(value: float) -> str:
    return f"{value:.2f}\\%"


def format_std(value: float) -> str:
    return f"{value:.2f}"


def write_latex_table(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_multiseed_main_table(summary: pd.DataFrame, report_tables: Path) -> None:
    metric_order = [
        ("center_accuracy", "Center Acc"),
        ("small_shift_accuracy", "Small Shift Acc"),
        ("large_shift_accuracy", "Large Shift Acc"),
        ("rotation_accuracy", "Rotation Acc"),
        ("shift_rotation_accuracy", "Shift+Rotation Acc"),
        ("grid_accuracy", "Grid Acc"),
        ("robust_drop", "Robust Drop"),
        ("rotation_drop", "Rotation Drop"),
    ]
    body = []
    for display_name, _ in MODELS:
        row = summary[summary["model"] == display_name].iloc[0]
        means = [format_percent(row[f"{metric}_mean"]) for metric, _ in metric_order]
        stds = [format_std(row[f"{metric}_std"]) for metric, _ in metric_order]
        body.append("      " + " & ".join([display_name, "Mean", *means]) + r" \\")
        body.append("      " + " & ".join(["", r"$\pm$ Std", *stds]) + r" \\")
    headers = " & ".join(["Model", "Stat", *[label for _, label in metric_order]])
    content = f"""\\begin{{table}}[H]
  \\centering
  \\caption{{不同模型的位置鲁棒性结果（五个随机种子汇总）。Mean 行给出五次运行均值，$\\pm$ Std 行给出跨种子标准差，标准差单位为百分点。}}
  \\label{{tab:main-results}}
  \\renewcommand{{\\arraystretch}}{{1.12}}
  \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{llcccccccc}}
      \\toprule
      {headers} \\\\
      \\midrule
{chr(10).join(body)}
      \\bottomrule
    \\end{{tabular}}%
  }}
\\end{{table}}
"""
    write_latex_table(report_tables / "main_results.tex", content)


def write_multiseed_ablation_table(summary: pd.DataFrame, table: pd.DataFrame, report_tables: Path) -> None:
    config_rows = {
        "MLP": ("center", "cls", "No"),
        "CNN": ("center", "cls", "No"),
        "ViT-AbsPos": ("center", "cls", "No"),
        "ViT-Aug": (r"shift\_rotation", "cls", "No"),
        "ViT-MeanPool": (r"shift\_rotation", "mean", "No"),
        "HybridConv-ViT": (r"shift\_rotation", "mean", "Yes"),
    }
    body = []
    for display_name, _ in MODELS:
        row = summary[summary["model"] == display_name].iloc[0]
        augmentation, pooling, conv_stem = config_rows[display_name]
        grid = f"{format_percent(row['grid_accuracy_mean'])} / {format_std(row['grid_accuracy_std'])}"
        drop = f"{format_percent(row['robust_drop_mean'])} / {format_std(row['robust_drop_std'])}"
        body.append(
            "      "
            + " & ".join([display_name, augmentation, pooling, conv_stem, grid, drop])
            + r" \\"
        )
    content = f"""\\begin{{table}}[H]
  \\centering
  \\caption{{模型消融实验结果（Mean / Std；Std 单位为百分点）}}
  \\label{{tab:ablation-results}}
  \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{lccccc}}
      \\toprule
      Model & Augmentation & Pooling & Conv Stem & Grid Acc & Robust Drop \\\\
      \\midrule
{chr(10).join(body)}
      \\bottomrule
    \\end{{tabular}}%
  }}
\\end{{table}}
"""
    write_latex_table(report_tables / "ablation_results.tex", content)


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
    axis.set_title("Grid Accuracy Across Seeds")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_metric_errorbars(summary: pd.DataFrame, output: Path) -> None:
    labels = [display_name for display_name, _ in MODELS]
    metric_labels = [label for _, label in METRICS]
    matrix = []
    for display_name in labels:
        row = summary[summary["model"] == display_name].iloc[0]
        matrix.append([row[f"{metric}_mean"] for metric, _ in METRICS])
    matrix = np.asarray(matrix, dtype=float)

    figure, axis = plt.subplots(figsize=(10.6, 5.4))
    image = axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(np.arange(len(metric_labels)), metric_labels)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Accuracy by Evaluation Protocol")
    axis.tick_params(axis="x", labelrotation=0, labelsize=10)
    axis.tick_params(axis="y", labelsize=10)
    axis.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value >= 58 else "#1F2933"
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold" if value >= 80 else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84, pad=0.02)
    colorbar.set_label("Accuracy (%)")
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
    all_deltas = []
    for left, right in pairs:
        deltas = []
        for seed in SEEDS:
            left_value = table[(table["model"] == left) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            right_value = table[(table["model"] == right) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            deltas.append((left_value - right_value) * 100)
        all_deltas.extend(deltas)
        label = f"{left} - {right}"
        axis.plot(seed_positions, deltas, marker="o", linewidth=2, label=label)
        for position, delta in zip(seed_positions, deltas):
            vertical_offset = 0.9 if delta >= 0 else -1.3
            axis.text(
                position,
                delta + vertical_offset,
                f"{delta:.1f}",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8,
            )
    axis.axhline(0, color="#222222", linewidth=1, linestyle="--")
    axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
    axis.set_ylim(min(-4.0, min(all_deltas) - 2.0), max(all_deltas) + 5.0)
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
        color="#244A73",
        label="Seven-angle mean",
        error_kw={"elinewidth": 1.3, "capsize": 3},
    )
    axis.barh(
        y + height / 2,
        summary["fixed_angle_worst_mean"],
        xerr=summary["fixed_angle_worst_std"],
        height=height,
        color="#C58B32",
        label="Worst scanned angle",
        error_kw={"elinewidth": 1.3, "capsize": 3},
    )
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 90)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title("Fixed-angle Robustness")
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
    report_tables = report_dir / "tables"
    report_figures = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = collect_records(root)
    summary = summarize(table)
    write_tables(table, summary, tables_dir)
    write_multiseed_main_table(summary, report_tables)
    write_multiseed_ablation_table(summary, table, report_tables)
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
