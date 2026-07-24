"""Create attention comparisons for selected edge-position samples.

The figure intentionally selects samples for which ViT-AbsPos is wrong while
HybridConv-ViT is correct.  It is an explanatory comparison, not a new metric:
attention rollout is shown only to inspect whether the two models focus on the
shifted foreground differently.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import CLASS_NAMES, build_test_dataset
from models import build_model
from utils.attention import attention_rollout, forward_with_attention
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import resolve_device, set_seed


DEFAULT_SHIFTS = ((-18, -18), (18, -18), (-18, 18), (18, 18))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ViT-AbsPos and Hybrid attention on corrected edge cases")
    parser.add_argument("--baseline-config", default="configs/seeds/vit_abspos_s42.yaml")
    parser.add_argument("--baseline-checkpoint", default="outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt")
    parser.add_argument("--hybrid-config", default="configs/seeds/hybrid_vit_s42.yaml")
    parser.add_argument("--hybrid-checkpoint", default="outputs/hybrid_vit_cpu_s42/checkpoints/best.pt")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument(
        "--shifts",
        default="-18,-18;18,-18;-18,18;18,18",
        help="Semicolon-separated dx,dy positions used to select edge cases",
    )
    parser.add_argument("--cases-per-figure", type=int, default=5)
    parser.add_argument("--output", default="outputs/figures/attention_abspos_vs_hybrid.png")
    parser.add_argument("--report-output", default="report/figures/attention_abspos_vs_hybrid.png")
    parser.add_argument("--manifest", default="outputs/tables/attention_abspos_wrong_hybrid_correct_samples.csv")
    return parser.parse_args()


def parse_shifts(raw: str) -> tuple[tuple[int, int], ...]:
    shifts = []
    for item in raw.split(";"):
        dx, dy = (int(value.strip()) for value in item.split(","))
        shifts.append((dx, dy))
    if not shifts:
        raise ValueError("At least one shift is required")
    return tuple(shifts)


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    config = load_config(config_path)
    model = build_model(config["model"]).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return config, model


@torch.no_grad()
def corrected_candidates(baseline, hybrid, dataset, device: torch.device, batch_size: int = 256):
    """Return rows where baseline is wrong and HybridConv-ViT is correct."""
    candidates = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    offset = 0
    for images, labels, meta in loader:
        images, labels = images.to(device), labels.to(device)
        baseline_logits, hybrid_logits = baseline(images), hybrid(images)
        baseline_pred = baseline_logits.argmax(dim=1)
        hybrid_pred = hybrid_logits.argmax(dim=1)
        corrected = (baseline_pred != labels) & (hybrid_pred == labels)
        baseline_confidence = F.softmax(baseline_logits, dim=1).amax(dim=1)
        hybrid_confidence = F.softmax(hybrid_logits, dim=1).amax(dim=1)
        for local_index in corrected.nonzero(as_tuple=False).flatten().cpu().tolist():
            candidates.append(
                {
                    "index": offset + local_index,
                    "label": int(labels[local_index].item()),
                    "baseline_prediction": int(baseline_pred[local_index].item()),
                    "hybrid_prediction": int(hybrid_pred[local_index].item()),
                    "baseline_confidence": float(baseline_confidence[local_index].item()),
                    "hybrid_confidence": float(hybrid_confidence[local_index].item()),
                    "dx": int(meta["dx"][local_index]),
                    "dy": int(meta["dy"][local_index]),
                }
            )
        offset += labels.shape[0]
    return candidates


def select_diverse_cases(candidates_by_shift, shifts, max_samples: int, seed: int):
    """Round-robin corners while preferring unseen labels and test indices."""
    rng = random.Random(seed)
    queues = {}
    for shift in shifts:
        queue = list(candidates_by_shift[shift])
        rng.shuffle(queue)
        queues[shift] = queue

    selected, used_labels, used_indices = [], set(), set()
    while len(selected) < max_samples:
        progressed = False
        for shift in shifts:
            if len(selected) >= max_samples:
                break
            queue = queues[shift]
            preferred = next(
                (row for row in queue if row["label"] not in used_labels and row["index"] not in used_indices),
                None,
            )
            fallback = next((row for row in queue if row["index"] not in used_indices), None)
            chosen = preferred or fallback
            if chosen is None:
                continue
            queue.remove(chosen)
            chosen = {**chosen, "shift": shift}
            selected.append(chosen)
            used_labels.add(chosen["label"])
            used_indices.add(chosen["index"])
            progressed = True
        if not progressed:
            break
    return selected


def overlay(axis, image, rollout):
    resized = F.interpolate(rollout[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    axis.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
    axis.imshow(resized, cmap="magma", vmin=0, vmax=1, alpha=0.58)
    axis.axis("off")


def write_manifest(path: Path, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "index", "dx", "dy", "true_label", "baseline_prediction", "baseline_confidence",
              "hybrid_prediction", "hybrid_confidence"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, case in enumerate(cases, start=1):
            writer.writerow(
                {
                    "case": number,
                    "index": case["index"],
                    "dx": case["dx"],
                    "dy": case["dy"],
                    "true_label": CLASS_NAMES[case["label"]],
                    "baseline_prediction": CLASS_NAMES[case["baseline_prediction"]],
                    "baseline_confidence": f"{case['baseline_confidence']:.4f}",
                    "hybrid_prediction": CLASS_NAMES[case["hybrid_prediction"]],
                    "hybrid_confidence": f"{case['hybrid_confidence']:.4f}",
                }
            )


def paged_path(path: Path, page: int, page_count: int) -> Path:
    if page_count == 1:
        return path
    return path.with_name(f"{path.stem}_{page:02d}{path.suffix}")


def render_page(cases, samples, baseline_maps, hybrid_maps, page: int, page_count: int, output: Path):
    # Keep the same canvas, spacing, raster quality and overlay treatment as
    # the project's single-model rollout figures.
    figure, axes = plt.subplots(len(cases), 3, figsize=(9, 3.1 * len(cases)), squeeze=False)
    for row, (case, sample, baseline_map, hybrid_map) in enumerate(zip(cases, samples, baseline_maps, hybrid_maps)):
        image, label, meta = sample
        raw = axes[row, 0]
        raw.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        raw.set_title(f"{CLASS_NAMES[label]}\ndx={meta['dx']}, dy={meta['dy']}", fontsize=8)
        raw.axis("off")
        overlay(axes[row, 1], image, baseline_map)
        overlay(axes[row, 2], image, hybrid_map)
        axes[row, 1].set_title("ViT-AbsPos", fontsize=9)
        axes[row, 2].set_title("HybridConv-ViT", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    shifts = parse_shifts(args.shifts)
    baseline_config = load_config(args.baseline_config)
    set_seed(baseline_config["seed"])
    device = resolve_device(baseline_config.get("device", "cpu"))
    _, baseline = load_model(args.baseline_config, args.baseline_checkpoint, device)
    hybrid_config, hybrid = load_model(args.hybrid_config, args.hybrid_checkpoint, device)
    if not (hasattr(baseline, "encoder") and hasattr(hybrid, "encoder")):
        raise ValueError("Both comparison models must be ViT-family models")

    datasets, candidates_by_shift = {}, defaultdict(list)
    for shift in shifts:
        dataset = build_test_dataset(baseline_config["data"], baseline_config["seed"] + 500, "grid_shift", shift)
        datasets[shift] = dataset
        candidates_by_shift[shift] = corrected_candidates(baseline, hybrid, dataset, device)
    cases = select_diverse_cases(candidates_by_shift, shifts, args.max_samples, baseline_config["seed"])
    if not cases:
        raise RuntimeError("No samples matched the criterion: ViT-AbsPos wrong and HybridConv-ViT correct")

    samples = [datasets[case["shift"]][case["index"]] for case in cases]
    images = torch.stack([sample[0] for sample in samples]).to(device)
    baseline_logits, baseline_layers = forward_with_attention(baseline, images)
    hybrid_logits, hybrid_layers = forward_with_attention(hybrid, images)
    baseline_maps = attention_rollout(baseline_layers, baseline.pooling).cpu()
    hybrid_maps = attention_rollout(hybrid_layers, hybrid.pooling).cpu()
    output, report_output, manifest = Path(args.output), Path(args.report_output), Path(args.manifest)
    if args.cases_per_figure < 1:
        raise ValueError("cases-per-figure must be positive")
    page_count = (len(cases) + args.cases_per_figure - 1) // args.cases_per_figure
    written = []
    for page, start in enumerate(range(0, len(cases), args.cases_per_figure), start=1):
        stop = start + args.cases_per_figure
        page_output = paged_path(output, page, page_count)
        render_page(cases[start:stop], samples[start:stop], baseline_maps[start:stop], hybrid_maps[start:stop], page, page_count, page_output)
        page_report_output = paged_path(report_output, page, page_count)
        page_report_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(page_output, page_report_output)
        written.append(page_output)
    write_manifest(manifest, cases)
    print(f"Selected {len(cases)} corrected cases across {len(shifts)} corners")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
