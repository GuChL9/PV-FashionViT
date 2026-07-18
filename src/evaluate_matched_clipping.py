from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from datasets import build_test_dataset
    from engine import evaluate
    from models import build_model
    from utils.checkpoint import load_checkpoint
    from utils.config import load_config
    from utils.seed import resolve_device, set_seed
except ModuleNotFoundError:  # Support importing as src.evaluate_matched_clipping.
    from src.datasets import build_test_dataset
    from src.engine import evaluate
    from src.models import build_model
    from src.utils.checkpoint import load_checkpoint
    from src.utils.config import load_config
    from src.utils.seed import resolve_device, set_seed


def matched_radius_coordinates(
    max_shift: int = 18,
    no_clip_limit: int = 14,
) -> list[dict[str, int | bool]]:
    """Return integer coordinates whose exact radius has both crop conditions."""

    grouped: dict[int, dict[bool, list[tuple[int, int]]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            clipped = abs(dx) > no_clip_limit or abs(dy) > no_clip_limit
            grouped[dx * dx + dy * dy][clipped].append((dx, dy))

    rows = []
    for radius_squared, conditions in sorted(grouped.items()):
        if not conditions[False] or not conditions[True]:
            continue
        for clipped in (False, True):
            for dx, dy in conditions[clipped]:
                rows.append(
                    {
                        "dx": dx,
                        "dy": dy,
                        "radius_squared": radius_squared,
                        "clipped": clipped,
                    }
                )
    return rows


def write_upsert(path: Path, rows: list[dict[str, object]], model: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing = [
                row
                for row in csv.DictReader(stream)
                if not (row.get("model") == model and int(row.get("seed", -1)) == seed)
            ]
    combined = [*existing, *rows]
    fieldnames = [
        "model",
        "seed",
        "dx",
        "dy",
        "radius_squared",
        "clipped",
        "accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)


def run_evaluation(args: argparse.Namespace) -> list[dict[str, object]]:
    config = load_config(args.config)
    config["seed"] = args.seed
    if args.run_name:
        config["run_name"] = args.run_name
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "cpu"))
    threads = int(config["train"].get("cpu_threads", 0) or 0)
    if device.type == "cpu" and threads:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(4, threads)))

    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["train"].get("label_smoothing", 0.0))
    data_cfg = config["data"]
    coordinates = matched_radius_coordinates(args.max_shift, args.no_clip_limit)
    rows = []
    for index, coordinate in enumerate(coordinates, start=1):
        dx, dy = int(coordinate["dx"]), int(coordinate["dy"])
        dataset = build_test_dataset(
            data_cfg,
            config["seed"] + 400,
            "grid_shift",
            fixed_shift=(dx, dy),
        )
        loader = DataLoader(
            dataset,
            batch_size=data_cfg.get("eval_batch_size", data_cfg["batch_size"]),
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=False,
        )
        result = evaluate(model, loader, criterion, device)
        rows.append(
            {
                "model": config["run_name"],
                "seed": config["seed"],
                **coordinate,
                "accuracy": result["accuracy"],
            }
        )
        if index % 8 == 0 or index == len(coordinates):
            print(f"{config['run_name']}: {index}/{len(coordinates)} matched coordinates")
    write_upsert(args.output, rows, config["run_name"], config["seed"])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact-distance clipping matches")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--max-shift", type=int, default=18)
    parser.add_argument("--no-clip-limit", type=int, default=14)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/matched_clipping_raw.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_evaluation(args)
    radii = sorted({int(row["radius_squared"]) for row in rows})
    print(f"Wrote {len(rows)} coordinates across {len(radii)} exact radii to {args.output}")


if __name__ == "__main__":
    main()
