from __future__ import annotations

import torch


def update_confusion_matrix(matrix, targets, predictions, num_classes: int = 10):
    indices = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    counts = torch.bincount(indices.cpu(), minlength=num_classes * num_classes)
    matrix += counts.reshape(num_classes, num_classes)
    return matrix


def per_class_accuracy(confusion_matrix):
    correct = confusion_matrix.diag().float()
    totals = confusion_matrix.sum(dim=1).float()
    return torch.where(totals > 0, correct / totals, torch.full_like(totals, float("nan")))


def robust_drop(center_accuracy: float, large_shift_accuracy: float) -> float:
    return float(center_accuracy - large_shift_accuracy)

