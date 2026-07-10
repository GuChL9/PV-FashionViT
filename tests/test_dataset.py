import torch
from torch.utils.data import Dataset

from src.datasets.pv_fashionmnist import PVFashionMNIST, ShiftRange


class ToyFashion(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        image = torch.zeros(1, 28, 28)
        image[:, 6:22, 8:20] = 1.0
        return image, index % 10


def test_dataset_shape_label_and_meta():
    dataset = PVFashionMNIST(ToyFashion(), mode="random_shift", seed=7)
    image, label, meta = dataset[3]
    assert len(dataset) == 20
    assert image.shape == (1, 56, 56)
    assert isinstance(label, int)
    assert {"dx", "dy", "scale", "angle"} <= set(meta)


def test_random_shift_is_reproducible_per_epoch():
    first = PVFashionMNIST(ToyFashion(), mode="random_shift", seed=7)
    second = PVFashionMNIST(ToyFashion(), mode="random_shift", seed=7)
    assert first[5][2] == second[5][2]
    first.set_epoch(1)
    assert first[5][2] != second[5][2]


def test_fixed_grid_shift_and_clipping():
    dataset = PVFashionMNIST(
        ToyFashion(), mode="grid_shift", fixed_shift=(18, -18), shift_range=ShiftRange(8, 18)
    )
    image, _, meta = dataset[0]
    assert (meta["dx"], meta["dy"]) == (18, -18)
    assert image.shape == (1, 56, 56)
    assert image.sum() > 0

