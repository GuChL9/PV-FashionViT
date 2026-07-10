from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class ShiftRange:
    small: int = 8
    large: int = 18


class PVFashionMNIST(Dataset):
    """Place FashionMNIST foregrounds on a larger canvas.

    Random parameters are derived from ``seed``, ``epoch`` and sample index.
    This avoids depending on DataLoader worker scheduling and keeps runs
    reproducible. Shifts beyond the no-clipping range are intentionally clipped
    at the canvas boundary.
    """

    MODES = {"center", "random_shift", "small_shift", "large_shift", "grid_shift", "affine"}

    def __init__(
        self,
        base_dataset: Dataset,
        canvas_size: int = 56,
        mode: str = "center",
        shift_range: ShiftRange | None = None,
        grid_values: Sequence[int] = (-18, -12, -6, 0, 6, 12, 18),
        fixed_shift: tuple[int, int] | None = None,
        affine_degrees: float = 10.0,
        affine_scale: tuple[float, float] = (0.9, 1.1),
        random_erasing_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(self.MODES)}")
        if canvas_size < 28:
            raise ValueError("canvas_size must be at least 28")
        if fixed_shift is not None and mode != "grid_shift":
            raise ValueError("fixed_shift is only valid for grid_shift mode")
        self.base_dataset = base_dataset
        self.canvas_size = int(canvas_size)
        self.mode = mode
        self.shift_range = shift_range or ShiftRange()
        self.grid_values = tuple(int(v) for v in grid_values)
        self.fixed_shift = fixed_shift
        self.affine_degrees = float(affine_degrees)
        self.affine_scale = tuple(float(v) for v in affine_scale)
        self.random_erasing_prob = float(random_erasing_prob)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        # Large odd constants mix the inputs without relying on salted hashes.
        mixed = (self.seed * 1_000_003 + self.epoch * 97_409 + index * 65_537) % (2**63 - 1)
        return torch.Generator().manual_seed(mixed)

    @staticmethod
    def _randint(low: int, high: int, generator: torch.Generator) -> int:
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    @staticmethod
    def _uniform(low: float, high: float, generator: torch.Generator) -> float:
        return float(torch.empty(1).uniform_(low, high, generator=generator).item())

    def _sample_parameters(self, index: int) -> tuple[int, int, float, float]:
        g = self._generator(index)
        angle, scale = 0.0, 1.0
        if self.mode == "center":
            dx = dy = 0
        elif self.mode == "small_shift":
            r = self.shift_range.small
            dx, dy = self._randint(-r, r, g), self._randint(-r, r, g)
        elif self.mode in {"random_shift", "large_shift", "affine"}:
            r = self.shift_range.large
            dx, dy = self._randint(-r, r, g), self._randint(-r, r, g)
            if self.mode == "affine":
                angle = self._uniform(-self.affine_degrees, self.affine_degrees, g)
                scale = self._uniform(self.affine_scale[0], self.affine_scale[1], g)
        else:
            if self.fixed_shift is not None:
                dx, dy = self.fixed_shift
            else:
                # Deterministically cycle through all grid positions.
                side = len(self.grid_values)
                dx = self.grid_values[index % side]
                dy = self.grid_values[(index // side) % side]
        return int(dx), int(dy), angle, scale

    def _place(self, image: Tensor, dx: int, dy: int) -> Tensor:
        _, height, width = image.shape
        top = (self.canvas_size - height) // 2 + dy
        left = (self.canvas_size - width) // 2 + dx
        canvas = torch.zeros((1, self.canvas_size, self.canvas_size), dtype=image.dtype)

        dst_top, dst_left = max(top, 0), max(left, 0)
        dst_bottom, dst_right = min(top + height, self.canvas_size), min(left + width, self.canvas_size)
        if dst_bottom <= dst_top or dst_right <= dst_left:
            return canvas
        src_top, src_left = dst_top - top, dst_left - left
        src_bottom = src_top + (dst_bottom - dst_top)
        src_right = src_left + (dst_right - dst_left)
        canvas[:, dst_top:dst_bottom, dst_left:dst_right] = image[:, src_top:src_bottom, src_left:src_right]
        return canvas

    def _erase(self, image: Tensor, generator: torch.Generator) -> Tensor:
        if self.random_erasing_prob <= 0:
            return image
        if self._uniform(0.0, 1.0, generator) >= self.random_erasing_prob:
            return image
        side = self._randint(4, 10, generator)
        top = self._randint(0, self.canvas_size - side, generator)
        left = self._randint(0, self.canvas_size - side, generator)
        return TF.erase(image, top, left, side, side, 0.0)

    def __getitem__(self, index: int) -> tuple[Tensor, int, dict[str, Any]]:
        image, label = self.base_dataset[index]
        image = TF.to_tensor(image) if not isinstance(image, Tensor) else image.float()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        dx, dy, angle, scale = self._sample_parameters(index)
        if angle != 0.0 or scale != 1.0:
            image = TF.affine(
                image,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        canvas = self._place(image, dx, dy)
        canvas = self._erase(canvas, self._generator(index + 10_000_019))
        meta = {"dx": dx, "dy": dy, "scale": scale, "angle": angle}
        return canvas, int(label), meta

