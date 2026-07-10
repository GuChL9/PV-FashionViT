"""Attention capture and rollout utilities for the project's ViT models.

PyTorch's ``TransformerEncoderLayer`` normally requests attention outputs with
``need_weights=False``.  The helper below temporarily asks its existing
MultiheadAttention modules for those weights, so it works with already trained
checkpoints without changing the model architecture or its state dict.
"""

from __future__ import annotations

import math

import torch


def _attention_modules(model):
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise TypeError("Attention rollout is available only for TinyViT and HybridConv-ViT models")
    return [layer.self_attn for layer in layers]


@torch.no_grad()
def forward_with_attention(model, images: torch.Tensor):
    """Run ``model`` and return logits plus per-layer attention weights.

    Each returned tensor has shape ``[batch, heads, tokens, tokens]``.
    """
    captured = []
    originals = []
    # TransformerEncoderLayer may use a fused inference path in eval mode.
    # That path bypasses MultiheadAttention.forward and exposes no weights.
    fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    for attention in _attention_modules(model):
        original = attention.forward

        def capture(*args, _original=original, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            output, weights = _original(*args, **kwargs)
            captured.append(weights.detach())
            return output, weights

        originals.append((attention, original))
        attention.forward = capture
    try:
        logits = model(images)
    finally:
        for attention, original in originals:
            attention.forward = original
        torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
    return logits, captured


def attention_rollout(attention_layers, pooling: str) -> torch.Tensor:
    """Fuse attention layers into one patch-level relevance map per sample."""
    if not attention_layers:
        raise ValueError("No attention layers were captured")
    batch, _, token_count, _ = attention_layers[0].shape
    joint = torch.eye(token_count, device=attention_layers[0].device).expand(batch, -1, -1)
    for layer_attention in attention_layers:
        matrix = layer_attention.mean(dim=1)
        matrix = matrix + torch.eye(token_count, device=matrix.device)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = matrix @ joint

    if pooling == "cls":
        relevance = joint[:, 0, 1:]
    else:
        relevance = joint.mean(dim=1)
    side = math.isqrt(relevance.shape[-1])
    if side * side != relevance.shape[-1]:
        raise ValueError("Patch-token count is not square")
    maps = relevance.reshape(batch, side, side)
    return maps / maps.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
