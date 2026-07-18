from __future__ import annotations

import time

import torch

try:
    from tqdm import tqdm
except ImportError:  # Keep lightweight CPU-only environments usable.
    class _PlainProgress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_kwargs):
            return None

    def tqdm(iterable, **_kwargs):
        return _PlainProgress(iterable)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, grad_clip: float | None = None):
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    loss_sum = 0.0
    correct = 0
    count = 0
    start = time.perf_counter()
    progress = tqdm(loader, desc=f"train {epoch + 1}", leave=False)
    for images, labels, _ in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += batch_size
        progress.set_postfix(loss=f"{loss_sum / count:.4f}", acc=f"{correct / count:.4f}")
    return {
        "train_loss": loss_sum / max(count, 1),
        "train_accuracy": correct / max(count, 1),
        "epoch_time": time.perf_counter() - start,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
