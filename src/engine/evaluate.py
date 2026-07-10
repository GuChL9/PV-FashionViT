from __future__ import annotations

import torch

from .metrics import per_class_accuracy, update_confusion_matrix


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 10, collect_predictions: bool = False):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    records = []
    offset = 0
    for images, labels, meta in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        count += batch_size
        update_confusion_matrix(confusion, labels, predictions, num_classes)
        if collect_predictions:
            labels_cpu, predictions_cpu = labels.cpu().tolist(), predictions.cpu().tolist()
            dx = meta.get("dx", [0] * batch_size)
            dy = meta.get("dy", [0] * batch_size)
            dx = dx.tolist() if hasattr(dx, "tolist") else dx
            dy = dy.tolist() if hasattr(dy, "tolist") else dy
            records.extend(
                {"index": offset + i, "target": y, "prediction": p, "dx": dx[i], "dy": dy[i]}
                for i, (y, p) in enumerate(zip(labels_cpu, predictions_cpu))
            )
        offset += batch_size
    class_acc = per_class_accuracy(confusion)
    return {
        "loss": loss_sum / max(count, 1),
        "accuracy": correct / max(count, 1),
        "per_class_accuracy": class_acc.tolist(),
        "confusion_matrix": confusion.tolist(),
        "predictions": records,
    }

