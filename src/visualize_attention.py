from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from datasets import CLASS_NAMES, build_test_dataset
from models import build_model
from utils.attention import attention_rollout, forward_with_attention
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate attention-rollout figures for a trained ViT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode", choices=["center", "grid_shift", "rotation", "shift_rotation"], default="center"
    )
    parser.add_argument("--dx", type=int, default=18, help="Used with --mode grid_shift")
    parser.add_argument("--dy", type=int, default=18, help="Used with --mode grid_shift")
    parser.add_argument("--angle", type=float, help="Fixed angle used with --mode rotation")
    parser.add_argument("--indices", default="0,1,2,3", help="Comma-separated test-set indices")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "cpu"))
    model = build_model(config["model"]).to(device)
    if not hasattr(model, "encoder"):
        raise ValueError("Attention rollout requires a ViT or HybridConv-ViT config")
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    fixed_shift = (args.dx, args.dy) if args.mode == "grid_shift" else None
    fixed_angle = args.angle if args.mode == "rotation" else None
    dataset = build_test_dataset(
        config["data"], config["seed"] + 300, args.mode,
        fixed_shift=fixed_shift, fixed_angle=fixed_angle,
    )
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    samples = [dataset[index] for index in indices]
    images = torch.stack([image for image, _, _ in samples]).to(device)
    labels = [label for _, label, _ in samples]
    metas = [meta for _, _, meta in samples]
    logits, layers = forward_with_attention(model, images)
    predictions = logits.argmax(dim=1).cpu().tolist()
    confidence = F.softmax(logits, dim=1).amax(dim=1).cpu().tolist()
    maps = attention_rollout(layers, getattr(model, "pooling", "cls")).cpu()

    figure, axes = plt.subplots(len(samples), 3, figsize=(9, 3.1 * len(samples)), squeeze=False)
    for row, (image, label, meta, prediction, score, rollout) in enumerate(
        zip(images.cpu(), labels, metas, predictions, confidence, maps)
    ):
        resized = F.interpolate(rollout[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        title = f"true: {CLASS_NAMES[label]}\npred: {CLASS_NAMES[prediction]} ({score:.2f})\n"
        title += f"dx={meta['dx']}, dy={meta['dy']}, angle={meta['angle']:.1f}°"
        axes[row, 0].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(title, fontsize=8)
        axes[row, 1].imshow(resized, cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_title("Attention rollout", fontsize=9)
        axes[row, 2].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(resized, cmap="magma", vmin=0, vmax=1, alpha=0.58)
        axes[row, 2].set_title("Overlay", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(f"{config['run_name']} — {args.mode}", y=1.01, fontsize=12)
    figure.tight_layout()
    angle_suffix = f"_{args.angle:+g}deg" if args.mode == "rotation" and args.angle is not None else ""
    output = Path(args.output) if args.output else (
        Path(config["output"]["save_dir"]) / config["run_name"] / "figures"
        / f"attention_rollout_{args.mode}{angle_suffix}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
