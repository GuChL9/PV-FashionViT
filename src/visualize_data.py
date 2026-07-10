from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision.transforms import functional as TF

from datasets.pv_fashionmnist import PVFashionMNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="outputs/figures")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = FashionMNIST(args.root, train=False, download=True)
    modes = ["original", "center", "random_shift", "grid_shift", "affine"]
    datasets = [None] + [PVFashionMNIST(base, mode=mode, seed=42) for mode in modes[1:]]

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

    positions = [(-18, -18), (0, -18), (18, -18), (-18, 0), (0, 0), (18, 0), (-18, 18), (0, 18), (18, 18)]
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for axis, (dx, dy) in zip(axes.flat, positions):
        dataset = PVFashionMNIST(base, mode="grid_shift", fixed_shift=(dx, dy), seed=42)
        image, _, _ = dataset[0]
        axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"dx={dx}, dy={dy}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "position_variation_demo.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
