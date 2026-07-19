"""Aggregate multi-seed metrics into tables and figures.

The script groups run directories by model and does not depend on PyTorch.
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
# Plot colors.
BLUE = "#1F77B4"
ORANGE = "#FF7F0E"
DARK = "#111111"
GRID = "#D9D9D9"
MODEL_PLOT_LABELS = ["MLP", "CNN", "ViT-\nAbsPos", "ViT-\nAug", "ViT-\nMeanPool", "Hybrid\nConv-ViT"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": DARK,
        "ytick.color": DARK,
        "text.color": DARK,
    }
)


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
            row[f"{metric}_q1"] = values.quantile(0.25)
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q3"] = values.quantile(0.75)
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
        grid = format_percent(row["grid_accuracy_median"])
        drop = format_percent(row["robust_drop_median"])
        body.append(
            "      "
            + " & ".join([display_name, augmentation, pooling, conv_stem, grid, drop])
            + r" \\"
        )
    content = f"""\\begin{{table}}[H]
  \\centering
  \\caption{{模型消融实验结果（五个随机种子的中位数）}}
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
    axis.set_facecolor("white")
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    axis.tick_params(axis="both", colors=DARK, labelsize=9.5, width=0.8, length=4)


def _draw_iqr(axis, x, medians, q1, q3, *, markersize: float = 4.5, capsize: float = 5.0) -> None:
    """Draw Q1--Q3 whiskers and a visible median marker above the bars."""
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack([medians - q1, q3 - medians]),
        fmt="o",
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=DARK,
        markeredgewidth=1.1,
        ecolor=DARK,
        elinewidth=1.55,
        capsize=capsize,
        capthick=1.55,
        linestyle="none",
        zorder=5,
    )


def _save_figure(figure, output: Path, *, dpi: int = 320) -> None:
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _label_x_axis(axis, labels: list[str], fontsize: int = 9, rotation: int = 0) -> None:
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.tick_params(axis="x", labelrotation=rotation, labelsize=fontsize, pad=5)


def plot_grid_errorbars(summary: pd.DataFrame, output: Path) -> None:
    labels = MODEL_PLOT_LABELS
    medians = summary["grid_accuracy_median"].to_numpy(dtype=float)
    q1 = summary["grid_accuracy_q1"].to_numpy(dtype=float)
    q3 = summary["grid_accuracy_q3"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10.0, 5.5), facecolor="white")
    bars = axis.bar(
        x,
        medians,
        width=0.56,
        color=BLUE,
        edgecolor=DARK,
        linewidth=0.55,
        zorder=3,
    )
    _draw_iqr(axis, x, medians, q1, q3, markersize=4.8, capsize=5.5)
    for bar, value, upper in zip(bars, medians, q3):
        axis.text(bar.get_x() + bar.get_width() / 2, upper + 1.2, f"{value:.1f}",
                  ha="center", va="bottom", fontsize=9.5, color=DARK)
    _label_x_axis(axis, labels, fontsize=9.5)
    axis.set_ylim(0, 90)
    axis.set_xlabel("Model", labelpad=8)
    axis.set_ylabel("Grid Accuracy (%)")
    axis.set_title("Grid Accuracy across Five Seeds", fontsize=13, fontweight="normal", pad=12)
    _style_axis(axis)
    figure.tight_layout(pad=1.0)
    _save_figure(figure, output)


def plot_metric_errorbars(summary: pd.DataFrame, output: Path) -> None:
    labels = MODEL_PLOT_LABELS
    panels = [
        ("center_accuracy", "Center Acc", "Accuracy (%)", BLUE),
        ("large_shift_accuracy", "Large Shift Acc", "Accuracy (%)", BLUE),
        ("grid_accuracy", "Grid Acc", "Accuracy (%)", BLUE),
        ("rotation_accuracy", "Rotation Acc", "Accuracy (%)", BLUE),
        ("shift_rotation_accuracy", "Shift+Rotation Acc", "Accuracy (%)", BLUE),
        ("robust_drop", "Robust Drop", "Percentage points", ORANGE),
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 3, figsize=(13.4, 7.6), sharex=True, facecolor="white")
    for axis, (metric, title, ylabel, color) in zip(axes.flat, panels):
        medians = summary[f"{metric}_median"].to_numpy(dtype=float)
        q1 = summary[f"{metric}_q1"].to_numpy(dtype=float)
        q3 = summary[f"{metric}_q3"].to_numpy(dtype=float)
        bars = axis.bar(
            x,
            medians,
            width=0.60,
            color=color,
            edgecolor=DARK,
            linewidth=0.45,
            zorder=3,
        )
        _draw_iqr(axis, x, medians, q1, q3, markersize=3.8, capsize=4.0)
        for bar, value, lower, upper in zip(bars, medians, q1, q3):
            text_y = upper + 1.0 if value >= 0 else lower - 1.0
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.4,
                color=DARK,
            )
        if metric == "robust_drop":
            axis.axhline(0, color=DARK, linewidth=0.9, zorder=2)
        axis.set_title(title, fontsize=11.5, fontweight="normal", pad=8)
        axis.set_ylabel(ylabel, fontsize=9.5)
        if metric == "robust_drop":
            q1 = summary[f"{metric}_q1"].min()
            q3 = summary[f"{metric}_q3"].max()
            axis.set_ylim(min(-6, q1 - 4), q3 + 6)
        else:
            axis.set_ylim(0, 96)
        _label_x_axis(axis, labels, fontsize=8.0)
        _style_axis(axis)
    figure.suptitle("Performance across Five Random Seeds", fontsize=14, fontweight="normal", y=0.985)
    figure.supxlabel("Model", fontsize=10.5, y=0.012)
    figure.tight_layout(rect=[0.01, 0.035, 0.995, 0.945], w_pad=1.5, h_pad=1.8)
    _save_figure(figure, output)


def plot_pairwise_deltas(table: pd.DataFrame, output: Path) -> None:
    pairs = [
        ("HybridConv-ViT", "ViT-MeanPool"),
        ("ViT-MeanPool", "ViT-Aug"),
        ("ViT-Aug", "ViT-AbsPos"),
    ]
    figure, axis = plt.subplots(figsize=(9.6, 5.3), facecolor="white")
    seed_positions = np.arange(len(SEEDS))
    all_deltas = []
    colors = [BLUE, ORANGE, "#2CA02C"]
    for (left, right), color in zip(pairs, colors):
        deltas = []
        for seed in SEEDS:
            left_value = table[(table["model"] == left) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            right_value = table[(table["model"] == right) & (table["seed"] == seed)]["grid_accuracy"].iloc[0]
            deltas.append((left_value - right_value) * 100)
        all_deltas.extend(deltas)
        label = f"{left} - {right}"
        axis.plot(
            seed_positions,
            deltas,
            marker="o",
            markersize=6.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=2.2,
            color=color,
            label=label,
            zorder=3,
        )
        for position, delta in zip(seed_positions, deltas):
            vertical_offset = 0.9 if delta >= 0 else -1.3
            axis.text(
                position,
                delta + vertical_offset,
                f"{delta:.1f}",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
            )
    axis.axhline(0, color=DARK, linewidth=1.0, linestyle="--", zorder=2)
    axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
    axis.set_ylim(min(-4.0, min(all_deltas) - 2.0), max(all_deltas) + 5.0)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Grid Accuracy Delta (percentage points)")
    axis.set_title("Paired Grid Accuracy Differences by Seed", fontsize=13, fontweight="normal", pad=12)
    axis.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="black", framealpha=0.95,
                loc="upper left")
    _style_axis(axis)
    figure.tight_layout(pad=1.0)
    _save_figure(figure, output)


def plot_angle_summary(summary: pd.DataFrame, output: Path) -> None:
    labels = MODEL_PLOT_LABELS
    x = np.arange(len(labels))
    width = 0.30
    figure, axis = plt.subplots(figsize=(10.4, 5.7), facecolor="white")
    mean_median = summary["fixed_angle_mean_median"].to_numpy(dtype=float)
    mean_q1 = summary["fixed_angle_mean_q1"].to_numpy(dtype=float)
    mean_q3 = summary["fixed_angle_mean_q3"].to_numpy(dtype=float)
    worst_median = summary["fixed_angle_worst_median"].to_numpy(dtype=float)
    worst_q1 = summary["fixed_angle_worst_q1"].to_numpy(dtype=float)
    worst_q3 = summary["fixed_angle_worst_q3"].to_numpy(dtype=float)
    bars_mean = axis.bar(
        x - width / 2,
        mean_median,
        width=width,
        color=BLUE,
        edgecolor=DARK,
        linewidth=0.5,
        label="Seven-angle mean",
        zorder=3,
    )
    bars_worst = axis.bar(
        x + width / 2,
        worst_median,
        width=width,
        color=ORANGE,
        edgecolor=DARK,
        linewidth=0.5,
        label="Worst scanned angle",
        zorder=3,
    )
    _draw_iqr(axis, x - width / 2, mean_median, mean_q1, mean_q3, markersize=4.2, capsize=4.5)
    _draw_iqr(axis, x + width / 2, worst_median, worst_q1, worst_q3, markersize=4.2, capsize=4.5)
    upper_quartiles = (mean_q3, worst_q3)
    for bars, q3_values in zip((bars_mean, bars_worst), upper_quartiles):
        for bar, upper in zip(bars, q3_values):
            value = bar.get_height()
            axis.text(bar.get_x() + bar.get_width() / 2, upper + 0.9, f"{value:.1f}",
                      ha="center", va="bottom", fontsize=9.0, color=DARK)
    _label_x_axis(axis, labels, fontsize=9.5)
    axis.set_ylim(0, 90)
    axis.set_xlabel("Model", labelpad=8)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Fixed-angle Robustness across Five Seeds", fontsize=13, fontweight="normal", pad=12)
    axis.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=0.95,
                loc="upper left", ncol=2, fontsize=9.5)
    _style_axis(axis)
    figure.tight_layout(pad=1.0)
    _save_figure(figure, output)


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
