from __future__ import annotations

from pathlib import Path

import matplotlib
# All figures are files, never interactive windows.  This keeps the training
# and analysis commands usable on Windows/CI installations without Tcl/Tk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: list[dict], output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    for key, val_key, filename, ylabel in [
        ("train_loss", "val_loss", "train_loss_curve.png", "Loss"),
        ("train_accuracy", "val_accuracy", "train_acc_curve.png", "Accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, [row[key] for row in history], marker="o", markersize=3, label="train")
        ax.plot(epochs, [row[val_key] for row in history], marker="o", markersize=3, label="validation")
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def plot_confusion_matrix(matrix, class_names, path) -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grid_heatmap(rows: list[dict], grid_values, path) -> None:
    lookup = {(int(row["dx"]), int(row["dy"])): float(row["accuracy"]) for row in rows}
    values = np.array([[lookup[(dx, dy)] for dx in grid_values] for dy in grid_values])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(image, ax=ax, label="Accuracy")
    ax.set(xticks=range(len(grid_values)), yticks=range(len(grid_values)),
           xticklabels=grid_values, yticklabels=grid_values, xlabel="dx", ylabel="dy")
    for iy in range(len(grid_values)):
        for ix in range(len(grid_values)):
            ax.text(ix, iy, f"{values[iy, ix]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_misclassified_examples(dataset, predictions: list[dict], class_names, path, limit: int = 16) -> None:
    mistakes = [row for row in predictions if row["target"] != row["prediction"]][:limit]
    if not mistakes:
        return
    columns = 4
    rows = (len(mistakes) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(10, 2.6 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, mistakes):
        image, _, meta = dataset[record["index"]]
        axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"true: {class_names[record['target']]}\npred: {class_names[record['prediction']]}\n"
            f"dx={meta['dx']}, dy={meta['dy']}",
            fontsize=8,
        )
        axis.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
