from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST

from .pv_fashionmnist import PVFashionMNIST, ShiftRange


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _split_train(base: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(base), generator=generator).tolist()
    val_size = int(round(len(base) * val_fraction))
    return Subset(base, order[val_size:]), Subset(base, order[:val_size])


def _pv(base: Dataset, cfg: dict[str, Any], mode: str, seed: int, fixed_shift=None) -> PVFashionMNIST:
    return PVFashionMNIST(
        base,
        canvas_size=cfg["canvas_size"],
        mode=mode,
        shift_range=ShiftRange(cfg["small_shift"], cfg["large_shift"]),
        grid_values=cfg["grid_values"],
        fixed_shift=fixed_shift,
        affine_degrees=cfg.get("affine_degrees", 10),
        affine_scale=tuple(cfg.get("affine_scale", [0.9, 1.1])),
        random_erasing_prob=cfg.get("random_erasing_prob", 0.0) if mode == cfg.get("train_mode") else 0.0,
        seed=seed,
    )


def build_dataloaders(cfg: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    root = Path(cfg["root"])
    base = FashionMNIST(root=root, train=True, download=True)
    train_base, val_base = _split_train(base, cfg.get("val_fraction", 0.1), seed)
    train_set = _pv(train_base, cfg, cfg["train_mode"], seed)
    val_set = _pv(val_base, cfg, cfg.get("val_mode", "center"), seed + 1)
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 2),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


def build_test_dataset(
    cfg: dict[str, Any],
    seed: int,
    mode: str = "center",
    fixed_shift: tuple[int, int] | None = None,
) -> PVFashionMNIST:
    base = FashionMNIST(root=Path(cfg["root"]), train=False, download=True)
    return _pv(base, cfg, mode, seed, fixed_shift=fixed_shift)

