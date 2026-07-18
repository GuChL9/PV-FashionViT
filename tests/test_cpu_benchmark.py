import torch

from src.benchmark_cpu import count_parameters, processor_name


def test_count_parameters_distinguishes_frozen_weights():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    model[1].weight.requires_grad = False

    total, trainable = count_parameters(model)

    assert total == 23
    assert trainable == 17


def test_processor_name_is_nonempty():
    assert processor_name()
