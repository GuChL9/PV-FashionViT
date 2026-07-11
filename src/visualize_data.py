from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision.transforms import functional as TF

from datasets.pv_fashionmnist import PVFashionMNIST, ShiftRange
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Override data.root from the selected config")
    parser.add_argument("--output", default="outputs/figures")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    data_cfg = config["data"]
    rotation_degrees = float(data_cfg["rotation_degrees"])
    angle_values = [float(value) for value in data_cfg["angle_values"]]
    dataset_kwargs = {
        "canvas_size": data_cfg["canvas_size"],
        "shift_range": ShiftRange(data_cfg["small_shift"], data_cfg["large_shift"]),
        "grid_values": data_cfg["grid_values"],
        "rotation_degrees": rotation_degrees,
        "affine_degrees": float(data_cfg.get("affine_degrees", rotation_degrees)),
        "affine_scale": tuple(data_cfg.get("affine_scale", [0.9, 1.1])),
        "seed": config["seed"],
    }
    base = FashionMNIST(args.root or data_cfg["root"], train=False, download=True)
    modes = ["original", "center", "random_shift", "rotation", "shift_rotation"]
    datasets = [None] + [
        PVFashionMNIST(base, mode=mode, **dataset_kwargs)
        for mode in modes[1:]
    ]

    fig, axes = plt.subplots(len(modes), 8, figsize=(12, 7.5))
    for row, (mode, dataset) in enumerate(zip(modes, datasets)):
        for col in range(8):
            if dataset is None:
                image, label = base[col]
                image = TF.to_tensor(image)
            else:
                image, label, _ = dataset[col]
            axes[row, col].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(mode, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "data_preview.png", dpi=180)
    plt.close(fig)

    edge = int(data_cfg["large_shift"])
    positions = [(-edge, -edge), (0, -edge), (edge, -edge), (-edge, 0), (0, 0), (edge, 0),
                 (-edge, edge), (0, edge), (edge, edge)]
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for axis, (dx, dy) in zip(axes.flat, positions):
        dataset = PVFashionMNIST(base, mode="grid_shift", fixed_shift=(dx, dy), **dataset_kwargs)
        image, _, _ = dataset[0]
        axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"dx={dx}, dy={dy}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "position_variation_demo.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(angle_values), figsize=(2.05 * len(angle_values), 2.4), squeeze=False)
    for axis, angle in zip(axes.flat, angle_values):
        dataset = PVFashionMNIST(
            base,
            mode="rotation",
            fixed_angle=angle,
            **dataset_kwargs,
        )
        image, _, _ = dataset[0]
        axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{angle:+g}°")
        axis.axis("off")
    fig.suptitle("Fixed-angle input sweep", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "angle_variation_demo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Visualize the configured perturbation protocols without iterating over the
    # training set.  This is a distribution audit, not a record of a past run.
    rng = np.random.default_rng(int(config["seed"]))
    sample_count = 12000
    shifts = rng.uniform(-edge, edge, size=(sample_count, 2))
    angles = rng.uniform(-rotation_degrees, rotation_degrees, size=sample_count)
    no_crop_limit = (int(data_cfg["canvas_size"]) - 28) / 2
    small_shift = float(data_cfg["small_shift"])

    fig, (ax_shift, ax_angle) = plt.subplots(
        1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    density = ax_shift.hist2d(
        shifts[:, 0], shifts[:, 1], bins=18,
        range=[[-edge, edge], [-edge, edge]], cmap="YlGnBu", cmin=1,
    )
    fig.colorbar(density[3], ax=ax_shift, fraction=0.046, pad=0.04, label="Samples per bin")
    for limit, color, style, label in [
        (no_crop_limit, "#C58B32", "--", "No-crop boundary"),
        (small_shift, "#4E8B66", ":", "Small-shift boundary"),
    ]:
        ax_shift.add_patch(plt.Rectangle(
            (-limit, -limit), 2 * limit, 2 * limit, fill=False,
            edgecolor=color, linewidth=1.8, linestyle=style, label=label,
        ))
    grid_values = np.asarray(data_cfg["grid_values"], dtype=float)
    gx, gy = np.meshgrid(grid_values, grid_values)
    ax_shift.scatter(gx, gy, s=13, c="#243B53", marker="o", alpha=0.75, label="49 fixed grid points")
    ax_shift.scatter([0], [0], s=95, c="#D64545", marker="*", zorder=5, label="Center")
    ax_shift.set(
        title="Translation coverage", xlabel="Horizontal shift dx (pixels)",
        ylabel="Vertical shift dy (pixels)", xlim=(-edge - 1, edge + 1),
        ylim=(-edge - 1, edge + 1), aspect="equal",
    )
    ax_shift.legend(loc="upper right", fontsize=8, frameon=True)

    bins = np.linspace(-rotation_degrees, rotation_degrees, 19)
    ax_angle.hist(angles, bins=bins, color="#2F7F83", alpha=0.82, edgecolor="white")
    ax_angle.axvline(0, color="#D64545", linewidth=1.8, label="Center angle")
    for index, angle in enumerate(angle_values):
        ax_angle.axvline(angle, color="#C58B32", linewidth=1.0, linestyle="--", alpha=0.8,
                         label="7 fixed test angles" if index == 0 else None)
    ax_angle.set(
        title="Rotation coverage", xlabel="Rotation angle (degrees)",
        ylabel="Sample count", xlim=(-rotation_degrees - 2, rotation_degrees + 2),
    )
    ax_angle.grid(axis="y", alpha=0.22)
    ax_angle.legend(loc="upper right", fontsize=8)
    ax_angle.text(
        0.02, 0.96,
        f"Configured random range: [{-rotation_degrees:g} degrees, {rotation_degrees:g} degrees]\n"
        f"Monte Carlo audit: n={sample_count:,}",
        transform=ax_angle.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88,
              "edgecolor": "#AAB7C4"},
    )
    fig.suptitle("Configured perturbation distributions and deterministic evaluation probes", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "perturbation_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
