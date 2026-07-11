from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    main()
