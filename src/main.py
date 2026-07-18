from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from datasets import CLASS_NAMES, build_dataloaders, build_test_dataset
from engine import evaluate, train_one_epoch
from engine.metrics import robust_drop
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.logger import ExperimentLogger
from utils.seed import resolve_device, set_seed
from utils.visualization import (
    plot_confusion_matrix,
    plot_grid_heatmap,
    plot_misclassified_examples,
    plot_training_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate PV-FashionViT models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="Override the random seed from the config")
    parser.add_argument("--run-name", help="Override run_name from the config")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--skip-grid", action="store_true", help="Skip the 49-position evaluation")
    parser.add_argument("--grid", action="store_true", help="Force the 49-position evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on a tiny subset")
    return parser.parse_args()


def make_optimizer(model, cfg):
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(optimizer, cfg):
    name = cfg.get("scheduler", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    if name in {"none", "off"}:
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def limited(loader, size: int):
    dataset = Subset(loader.dataset, range(min(size, len(loader.dataset))))
    return DataLoader(dataset, batch_size=min(loader.batch_size, 16), shuffle=False, num_workers=0)


def test_loader(dataset, cfg, smoke: bool = False):
    if smoke:
        dataset = Subset(dataset, range(min(32, len(dataset))))
    eval_batch_size = cfg.get("eval_batch_size", cfg["batch_size"])
    return DataLoader(dataset, batch_size=min(eval_batch_size, 32) if smoke else eval_batch_size,
                      shuffle=False, num_workers=0 if smoke else cfg.get("num_workers", 2),
                      pin_memory=torch.cuda.is_available())


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv(path, key, row, fieldnames):
    path = Path(path)
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = [old for old in csv.DictReader(stream) if old.get(key) != str(row[key])]
    rows.append(row)
    write_csv(path, rows, fieldnames)


def run_evaluation_suite(
    model,
    criterion,
    device,
    config,
    run_dir: Path,
    smoke: bool,
    skip_grid: bool,
    publish_global: bool = True,
):
    data_cfg, seed = config["data"], config["seed"]
    conditions = {}
    large_shift_dataset = None
    evaluation_modes = [
        ("center", "center"),
        ("small_shift", "small_shift"),
        ("large_shift", "large_shift"),
        ("rotation", "rotation"),
        ("shift_rotation", "shift_rotation"),
    ]
    for name, mode in evaluation_modes:
        dataset = build_test_dataset(data_cfg, seed + 100, mode)
        if name == "large_shift":
            large_shift_dataset = dataset
        conditions[name] = evaluate(
            model,
            test_loader(dataset, data_cfg, smoke),
            criterion,
            device,
            collect_predictions=(name == "large_shift"),
        )

    grid_rows = []
    angle_rows = []
    if not skip_grid:
        grid_values = [0] if smoke else data_cfg["grid_values"]
        for dy in grid_values:
            for dx in grid_values:
                dataset = build_test_dataset(data_cfg, seed + 200, "grid_shift", (dx, dy))
                result = evaluate(model, test_loader(dataset, data_cfg, smoke), criterion, device)
                grid_rows.append({"model": config["run_name"], "dx": dx, "dy": dy, "accuracy": result["accuracy"]})
        write_csv(run_dir / "tables" / "grid_accuracy.csv", grid_rows)
        plot_grid_heatmap(grid_rows, grid_values, run_dir / "figures" / "grid_accuracy_heatmap.png")

        angle_values = [0] if smoke else data_cfg["angle_values"]
        for angle in angle_values:
            dataset = build_test_dataset(data_cfg, seed + 300, "rotation", fixed_angle=angle)
            result = evaluate(model, test_loader(dataset, data_cfg, smoke), criterion, device)
            angle_rows.append({"model": config["run_name"], "angle": angle, "accuracy": result["accuracy"]})
        write_csv(run_dir / "tables" / "angle_accuracy.csv", angle_rows)

    grid_accuracy = sum(row["accuracy"] for row in grid_rows) / len(grid_rows) if grid_rows else float("nan")
    summary = {
        "model": config["run_name"],
        "center_accuracy": conditions["center"]["accuracy"],
        "small_shift_accuracy": conditions["small_shift"]["accuracy"],
        "large_shift_accuracy": conditions["large_shift"]["accuracy"],
        "rotation_accuracy": conditions["rotation"]["accuracy"],
        "shift_rotation_accuracy": conditions["shift_rotation"]["accuracy"],
        "grid_accuracy": grid_accuracy,
        "robust_drop": robust_drop(conditions["center"]["accuracy"], conditions["large_shift"]["accuracy"]),
        "rotation_drop": robust_drop(conditions["center"]["accuracy"], conditions["rotation"]["accuracy"]),
    }
    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as stream:
        json.dump({"summary": summary, "conditions": conditions}, stream, ensure_ascii=False, indent=2)
    write_csv(run_dir / "tables" / "predictions.csv", conditions["large_shift"]["predictions"])
    write_csv(
        run_dir / "tables" / "per_class_accuracy.csv",
        [{"class": name, "accuracy": acc} for name, acc in zip(CLASS_NAMES, conditions["large_shift"]["per_class_accuracy"])],
    )
    plot_confusion_matrix(
        conditions["large_shift"]["confusion_matrix"],
        CLASS_NAMES,
        run_dir / "figures" / "confusion_matrix.png",
    )
    if large_shift_dataset is not None:
        plot_misclassified_examples(
            large_shift_dataset,
            conditions["large_shift"]["predictions"],
            CLASS_NAMES,
            run_dir / "figures" / "misclassified_examples.png",
        )

    # A smoke run validates the code path only.  It must remain completely
    # separate from formal experiment summaries used in plots and reports.
    if not publish_global:
        return summary

    global_tables = Path(config["output"]["save_dir"]) / "tables"
    upsert_csv(
        global_tables / "main_results.csv",
        "model",
        summary,
        ["model", "center_accuracy", "small_shift_accuracy", "large_shift_accuracy",
         "rotation_accuracy", "shift_rotation_accuracy", "grid_accuracy", "robust_drop", "rotation_drop"],
    )
    ablation_row = {
        "model": config["run_name"],
        "augmentation": config["data"]["train_mode"],
        "pooling": config["model"].get("pooling", "n/a"),
        "conv_stem": config["model"]["name"] == "hybrid_vit",
        "grid_accuracy": grid_accuracy,
        "robust_drop": summary["robust_drop"],
    }
    upsert_csv(
        global_tables / "ablation_results.csv",
        "model",
        ablation_row,
        ["model", "augmentation", "pooling", "conv_stem", "grid_accuracy", "robust_drop"],
    )
    per_class_path = global_tables / "per_class_accuracy.csv"
    per_class_rows = {}
    model_columns = []
    if per_class_path.exists():
        with per_class_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            model_columns = [name for name in (reader.fieldnames or []) if name != "class"]
            per_class_rows = {row["class"]: row for row in reader if row.get("class")}
    if config["run_name"] not in model_columns:
        model_columns.append(config["run_name"])
    for class_name, accuracy in zip(CLASS_NAMES, conditions["large_shift"]["per_class_accuracy"]):
        per_class_rows.setdefault(class_name, {"class": class_name})[config["run_name"]] = accuracy
    write_csv(per_class_path, [per_class_rows[name] for name in CLASS_NAMES], ["class", *model_columns])
    if grid_rows:
        existing = []
        global_grid = global_tables / "grid_accuracy.csv"
        if global_grid.exists():
            with global_grid.open("r", encoding="utf-8", newline="") as stream:
                existing = [row for row in csv.DictReader(stream) if row.get("model") != config["run_name"]]
        write_csv(global_grid, existing + grid_rows, ["model", "dx", "dy", "accuracy"])
    if angle_rows:
        existing = []
        global_angles = global_tables / "angle_accuracy.csv"
        if global_angles.exists():
            with global_angles.open("r", encoding="utf-8", newline="") as stream:
                existing = [row for row in csv.DictReader(stream) if row.get("model") != config["run_name"]]
        write_csv(global_angles, existing + angle_rows, ["model", "angle", "accuracy"])
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.run_name:
        config["run_name"] = args.run_name
    if args.smoke:
        config["train"]["epochs"] = 1
        config["run_name"] += "_smoke"
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    cpu_threads = int(config["train"].get("cpu_threads", 0) or 0)
    if device.type == "cpu" and cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    run_dir = Path(config["output"]["save_dir"]) / config["run_name"]
    for name in ["checkpoints", "logs", "figures", "tables"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["train"].get("label_smoothing", 0.0))
    optimizer = make_optimizer(model, config["train"])
    scheduler = make_scheduler(optimizer, config["train"])
    best_path = run_dir / "checkpoints" / "best.pt"

    if args.eval_only:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else best_path
        load_checkpoint(checkpoint_path, model, map_location=device)
    else:
        train_loader, val_loader = build_dataloaders(config["data"], config["seed"])
        if args.smoke:
            train_loader, val_loader = limited(train_loader, 64), limited(val_loader, 32)
        logger = ExperimentLogger(run_dir / "logs")
        history = []
        best_acc = -1.0
        epochs_without_improvement = 0
        patience = int(config["train"].get("early_stopping_patience", 0) or 0)
        min_epochs = int(config["train"].get("early_stopping_min_epochs", 0) or 0)
        min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
        for epoch in range(config["train"]["epochs"]):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                config["train"].get("grad_clip"),
            )
            validation = evaluate(model, val_loader, criterion, device)
            row = {"epoch": epoch + 1, **train_metrics, "val_loss": validation["loss"],
                   "val_accuracy": validation["accuracy"]}
            history.append(row)
            logger.log(row)
            if validation["accuracy"] > best_acc + min_delta:
                best_acc = validation["accuracy"]
                epochs_without_improvement = 0
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_acc, config)
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step()
            save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, best_acc, config)
            print(json.dumps(row, ensure_ascii=False))
            if patience > 0 and epoch + 1 >= min_epochs and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}; best validation accuracy={best_acc:.4f}")
                break
        plot_training_curves(history, run_dir / "figures")
        load_checkpoint(best_path, model, map_location=device)

    run_grid_by_default = bool(config.get("evaluation", {}).get("run_grid_after_training", False))
    skip_grid = args.skip_grid or (not args.grid and not run_grid_by_default)
    summary = run_evaluation_suite(
        model,
        criterion,
        device,
        config,
        run_dir,
        args.smoke,
        skip_grid,
        publish_global=(
            not args.smoke and config.get("output", {}).get("publish_global", True)
        ),
    )
    print(json.dumps({"device": str(device), "cpu_threads": torch.get_num_threads(), **summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
