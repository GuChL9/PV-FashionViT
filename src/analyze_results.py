"""Build representative report assets from the seed-42 multi-seed runs.

The five-seed tables and uncertainty figures are produced separately by
``analyze_multiseed_results.py``.  This script keeps the detailed plots that
need one concrete run (training curves, spatial grids, angle scans and class
results) without requiring a second set of unsuffixed single-seed models.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DISPLAY_NAMES = {
    "mlp_center_cpu_s42": "MLP",
    "cnn_center_cpu_s42": "CNN",
    "vit_abspos_center_cpu_s42": "ViT-AbsPos",
    "vit_aug_cpu_s42": "ViT-Aug",
    "vit_meanpool_cpu_s42": "ViT-MeanPool",
    "hybrid_vit_cpu_s42": "HybridConv-ViT",
}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create formal result tables and report figures")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--report-dir", default="report")
    return parser.parse_args()


def _display_name(run_name: str) -> str:
    return DISPLAY_NAMES.get(run_name, run_name.replace("_", " "))


def discover_runs(root: Path):
    """Read exactly the six representative seed-42 formal runs."""
    records = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_dir.name.endswith("_smoke"):
            continue
        evaluation_path = run_dir / "evaluation.json"
        config_path = run_dir / "config.yaml"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not (evaluation_path.exists() and config_path.exists() and checkpoint.exists()):
            continue
        with evaluation_path.open(encoding="utf-8") as stream:
            evaluation = json.load(stream)
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        summary = evaluation.get("summary", {})
        if not summary:
            continue
        run_name = config.get("run_name", run_dir.name)
        if run_name not in DISPLAY_NAMES:
            continue
        records.append({"run_dir": run_dir, "config": config, "evaluation": evaluation, "summary": summary,
                        "run_name": run_name, "display_name": _display_name(run_name)})
    order = {name: index for index, name in enumerate(DISPLAY_NAMES)}
    return sorted(records, key=lambda row: (order.get(row["run_name"], len(order)), row["run_name"]))


def to_percent(value) -> str:
    return "--" if value is None or not math.isfinite(float(value)) else f"{100 * float(value):.2f}\\%"


def write_latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]):
    alignment = "l" + "c" * (len(headers) - 1)
    body = "\n".join("      " + " & ".join(row) + r" \\" for row in rows)
    content = f"""\\begin{{table}}[H]
  \\centering
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{{alignment}}}
      \\toprule
      {' & '.join(headers)} \\\\
      \\midrule
{body}
      \\bottomrule
    \\end{{tabular}}%
  }}
\\end{{table}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def bar_chart(table: pd.DataFrame, column: str, title: str, ylabel: str, output: Path):
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    values = table[column] * 100
    bars = axis.bar(table["display_name"], values, color="#3B82B6")
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def annotation_color(cmap, value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Choose the higher-contrast black/white label for a heatmap cell."""
    rgba = plt.get_cmap(cmap)(plt.Normalize(vmin=vmin, vmax=vmax)(value))
    rgb = np.asarray(rgba[:3])
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear_rgb, [0.2126, 0.7152, 0.0722]))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def grid_panel(records, output: Path):
    count = len(records)
    columns = 3
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14.2, 9.4),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for axis in axes.flat[count:]:
        axis.set_axis_off()
    for axis, record in zip(axes.flat, records):
        axis.set_axis_on()
        grid_path = record["run_dir"] / "tables" / "grid_accuracy.csv"
        if not grid_path.exists():
            axis.text(0.5, 0.5, "Grid evaluation missing", ha="center", va="center")
            axis.set_title(record["display_name"], pad=10)
            continue
        grid = pd.read_csv(grid_path)
        values = sorted(grid["dx"].unique())
        lookup = {(int(row.dx), int(row.dy)): float(row.accuracy) for row in grid.itertuples()}
        matrix = np.array([[lookup[(dx, dy)] for dx in values] for dy in values])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set(title=record["display_name"], xlabel="dx", ylabel="dy",
                 xticks=range(len(values)), yticks=range(len(values)), xticklabels=values, yticklabels=values)
        axis.set_title(record["display_name"], pad=10)
        axis.xaxis.labelpad = 2
        axis.yaxis.labelpad = 2
        for iy, row in enumerate(matrix):
            for ix, value in enumerate(row):
                axis.text(
                    ix,
                    iy,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=annotation_color("viridis", value),
                    fontsize=7,
                )
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Accuracy", shrink=0.9, pad=0.02)
    figure.suptitle("Position-grid accuracy across formal experiments", fontsize=16)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def training_overview(records, output: Path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for record in records:
        history_path = record["run_dir"] / "logs" / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        epochs = pd.to_numeric(history["epoch"], errors="coerce").to_numpy(dtype=float)
        restarts = np.flatnonzero(np.diff(epochs) <= 0) + 1
        if len(restarts):
            history = history.iloc[restarts[-1]:].copy()
        axes[0].plot(history["epoch"], history["val_accuracy"], linewidth=2, label=record["display_name"])
        axes[1].plot(history["epoch"], history["val_loss"], linewidth=2, label=record["display_name"])
    axes[0].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def angle_robustness_heatmap(records, output: Path):
    """Plot fixed-angle accuracy as concentric fan sectors plus mean bars."""
    tables, observed_angles = [], set()
    for record in records:
        angle_path = record["run_dir"] / "tables" / "angle_accuracy.csv"
        if not angle_path.exists():
            continue
        table = pd.read_csv(angle_path).sort_values("angle")
        if table.empty:
            continue
        observed_angles.update(table["angle"].to_numpy(dtype=float))
        tables.append((record, table.set_index("angle")))
    if not tables:
        return
    angles = sorted(observed_angles)
    matrix, labels = [], []
    for record, table in tables:
        values = np.array([float(table.loc[angle, "accuracy"]) for angle in angles]) * 100
        matrix.append(values)
        labels.append(record["display_name"])
    matrix = np.asarray(matrix)
    means = matrix.mean(axis=1)
    angle_array = np.asarray(angles, dtype=float)
    midpoints = (angle_array[:-1] + angle_array[1:]) / 2
    theta_edges = np.concatenate((
        [angle_array[0] - (midpoints[0] - angle_array[0])],
        midpoints,
        [angle_array[-1] + (angle_array[-1] - midpoints[-1])],
    ))
    theta_edges = np.deg2rad(theta_edges)
    inner_radius = 3.0
    radial_edges = inner_radius + np.arange(len(labels) + 1, dtype=float)

    figure = plt.figure(figsize=(12, 9.2), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=[4.8, 1.35])
    axis = figure.add_subplot(grid[0], projection="polar")
    mean_axis = figure.add_subplot(grid[1])
    image = axis.pcolormesh(theta_edges, radial_edges, matrix, vmin=0, vmax=100,
                            cmap="YlGnBu", edgecolors="white", linewidth=2, shading="flat")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_thetamin(float(np.rad2deg(theta_edges[0])))
    axis.set_thetamax(float(np.rad2deg(theta_edges[-1])))
    axis.set_xticks(np.deg2rad(angle_array), [f"{angle:+g}°" for angle in angle_array])
    axis.set_yticks(inner_radius + np.arange(len(labels)) + 0.5, labels)
    axis.set_rlabel_position(-58)
    axis.tick_params(axis="y", pad=8, labelsize=9)
    axis.grid(False)
    axis.spines["polar"].set_visible(False)
    axis.set_title("Rotation accuracy fan", fontsize=16, fontweight="bold", pad=22)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(np.deg2rad(angle_array[column]), inner_radius + row + 0.5, f"{value:.0f}",
                      ha="center", va="center", fontsize=7.5,
                      color=annotation_color("YlGnBu", value, 0, 100))
    colorbar = figure.colorbar(image, ax=axis, pad=0.08, shrink=0.72)
    colorbar.set_label("Accuracy (%)")

    normalization = plt.Normalize(0, 100)
    colors = plt.get_cmap("YlGnBu")(normalization(means))
    bars = mean_axis.bar(labels, means, color=colors, edgecolor="white", linewidth=1.2)
    for bar, value in zip(bars, means):
        mean_axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%",
                       ha="center", va="bottom", fontsize=9, fontweight="bold")
    mean_axis.set_ylim(0, 100)
    mean_axis.set_ylabel("Mean accuracy (%)")
    mean_axis.set_title("Average across seven angles", fontsize=12, pad=8)
    mean_axis.grid(axis="y", alpha=0.2)
    mean_axis.tick_params(axis="x", rotation=15)
    mean_axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def per_class_heatmap(per_class: pd.DataFrame, output: Path):
    matrix = per_class.drop(columns="class").to_numpy(dtype=float).T
    figure, axis = plt.subplots(figsize=(11.5, 4.8))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    figure.colorbar(image, ax=axis, label="Large-shift accuracy")
    axis.set(yticks=range(len(per_class.columns) - 1), yticklabels=per_class.columns[1:],
             xticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, xlabel="Class", ylabel="Model")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for iy, row in enumerate(matrix):
        for ix, value in enumerate(row):
            axis.text(ix, iy, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                      color=annotation_color("YlGnBu", value))
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def copy_if_present(source: Path, destination: Path):
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def grid_accuracy_from_run(record) -> float:
    """Prefer the concrete 49-row CSV over an older summary JSON value."""
    grid_path = record["run_dir"] / "tables" / "grid_accuracy.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        if len(grid) == 49 and "accuracy" in grid:
            return float(grid["accuracy"].mean())
    return float(record["summary"].get("grid_accuracy", float("nan")))


def main():
    args = parse_args()
    root, report_dir = Path(args.root), Path(args.report_dir)
    tables_dir, figures_dir = root / "tables", root / "figures"
    report_tables, report_figures = report_dir / "tables", report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    records = discover_runs(root)
    if not records:
        raise FileNotFoundError("No complete seed-42 multi-seed outputs were found")
    found = {record["run_name"] for record in records}
    missing = [run_name for run_name in DISPLAY_NAMES if run_name not in found]
    if missing:
        raise FileNotFoundError("Missing representative seed-42 runs: " + ", ".join(missing))

    main_rows, ablation_rows, per_class = [], [], {name: {"class": name} for name in CLASS_NAMES}
    all_grid_rows, all_angle_rows = [], []
    for record in records:
        summary, config = record["summary"], record["config"]
        grid_accuracy = grid_accuracy_from_run(record)
        row = {"model": record["run_name"], "display_name": record["display_name"],
               "center_accuracy": summary["center_accuracy"], "small_shift_accuracy": summary["small_shift_accuracy"],
               "large_shift_accuracy": summary["large_shift_accuracy"], "grid_accuracy": grid_accuracy,
               "rotation_accuracy": summary.get("rotation_accuracy", float("nan")),
               "shift_rotation_accuracy": summary.get("shift_rotation_accuracy", float("nan")),
               "robust_drop": summary["robust_drop"],
               "rotation_drop": summary.get("rotation_drop", float("nan"))}
        main_rows.append(row)
        ablation_rows.append({"model": record["run_name"], "display_name": record["display_name"],
                               "augmentation": config["data"]["train_mode"], "pooling": config["model"].get("pooling", "n/a"),
                               "conv_stem": config["model"]["name"] == "hybrid_vit", "grid_accuracy": grid_accuracy,
                               "robust_drop": summary["robust_drop"]})
        large = record["evaluation"]["conditions"]["large_shift"]["per_class_accuracy"]
        for class_name, accuracy in zip(CLASS_NAMES, large):
            per_class[class_name][record["display_name"]] = accuracy
        grid_path = record["run_dir"] / "tables" / "grid_accuracy.csv"
        if grid_path.exists():
            grid = pd.read_csv(grid_path)
            all_grid_rows.append(grid.assign(model=record["run_name"], display_name=record["display_name"]))
        angle_path = record["run_dir"] / "tables" / "angle_accuracy.csv"
        if angle_path.exists():
            angles = pd.read_csv(angle_path)
            all_angle_rows.append(angles.assign(model=record["run_name"], display_name=record["display_name"]))

    main_table = pd.DataFrame(main_rows)
    ablation_table = pd.DataFrame(ablation_rows)
    per_class_table = pd.DataFrame([per_class[name] for name in CLASS_NAMES])
    main_table.drop(columns="display_name").to_csv(tables_dir / "main_results.csv", index=False)
    ablation_table.drop(columns="display_name").to_csv(tables_dir / "ablation_results.csv", index=False)
    per_class_table.to_csv(tables_dir / "per_class_accuracy.csv", index=False)
    if all_grid_rows:
        pd.concat(all_grid_rows, ignore_index=True).drop(columns="display_name").to_csv(tables_dir / "grid_accuracy.csv", index=False)
    if all_angle_rows:
        pd.concat(all_angle_rows, ignore_index=True).drop(columns="display_name").to_csv(
            tables_dir / "angle_accuracy.csv", index=False
        )

    write_latex_table(
        report_tables / "main_results.tex", "不同模型的位置鲁棒性结果", "tab:main-results",
        ["Model", "Center Acc", "Small Shift Acc", "Large Shift Acc", "Rotation Acc",
         "Shift+Rotation Acc", "Grid Acc", "Robust Drop", "Rotation Drop"],
        [[row.display_name, to_percent(row.center_accuracy), to_percent(row.small_shift_accuracy),
          to_percent(row.large_shift_accuracy), to_percent(row.rotation_accuracy),
          to_percent(row.shift_rotation_accuracy), to_percent(row.grid_accuracy),
          to_percent(row.robust_drop), to_percent(row.rotation_drop)]
         for row in main_table.itertuples()],
    )
    write_latex_table(
        report_tables / "ablation_results.tex", "模型消融实验结果", "tab:ablation-results",
        ["Model", "Augmentation", "Pooling", "Conv Stem", "Grid Acc", "Robust Drop"],
        [[row.display_name, str(row.augmentation).replace("_", r"\_"), row.pooling,
          "Yes" if row.conv_stem else "No",
          to_percent(row.grid_accuracy), to_percent(row.robust_drop)] for row in ablation_table.itertuples()],
    )
    write_latex_table(
        report_tables / "per_class_accuracy.tex", "Large Shift 条件下的类别准确率", "tab:per-class-results",
        list(per_class_table.columns),
        [[row["class"], *[to_percent(row[column]) for column in per_class_table.columns[1:]]]
         for _, row in per_class_table.iterrows()],
    )

    bar_chart(main_table, "center_accuracy", "Center accuracy by model", "Accuracy (%)", figures_dir / "model_accuracy_comparison.png")
    bar_chart(main_table, "robust_drop", "Robust drop by model", "Center - large shift (pp)", figures_dir / "robust_drop_comparison.png")
    bar_chart(main_table, "grid_accuracy", "Grid accuracy by model", "Accuracy (%)", figures_dir / "grid_accuracy_comparison.png")
    bar_chart(main_table, "rotation_accuracy", "Rotation accuracy by model", "Accuracy (%)", figures_dir / "rotation_accuracy_comparison.png")
    angle_heatmap = figures_dir / "angle_robustness_heatmap.png"
    angle_robustness_heatmap(records, angle_heatmap)
    copy_if_present(angle_heatmap, figures_dir / "angle_robustness_polar.png")
    grid_panel(records, figures_dir / "grid_accuracy_panel.png")
    training_overview(records, figures_dir / "training_curves_overview.png")
    per_class_heatmap(per_class_table, figures_dir / "per_class_accuracy_heatmap.png")

    for name in ["data_preview.png", "position_variation_demo.png", "angle_variation_demo.png",
                 "perturbation_distribution.png",
                 "model_accuracy_comparison.png", "robust_drop_comparison.png",
                 "grid_accuracy_comparison.png", "rotation_accuracy_comparison.png", "angle_robustness_heatmap.png",
                 "angle_robustness_polar.png", "grid_accuracy_panel.png",
                 "training_curves_overview.png", "per_class_accuracy_heatmap.png"]:
        copy_if_present(figures_dir / name, report_figures / name)
    final_record = max(records, key=lambda row: row["summary"]["large_shift_accuracy"])
    for name in ["confusion_matrix.png", "misclassified_examples.png", "grid_accuracy_heatmap.png", "attention_rollout_center.png", "attention_rollout_grid_shift.png"]:
        copy_if_present(final_record["run_dir"] / "figures" / name, report_figures / f"final_{name}")
    print(f"Prepared {len(records)} formal runs; best large-shift model: {final_record['display_name']}")


if __name__ == "__main__":
    main()
