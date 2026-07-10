from __future__ import annotations

from .cnn import CNNClassifier
from .hybrid_vit import HybridConvViT
from .mlp import MLPClassifier
from .vit import PatchEmbedding, TinyViT


def build_model(cfg):
    name = cfg["name"]
    common = {
        "img_size": cfg.get("img_size", 56),
        "in_channels": cfg.get("in_channels", 1),
        "num_classes": cfg.get("num_classes", 10),
        "dropout": cfg.get("dropout", 0.1),
    }
    if name == "mlp":
        return MLPClassifier(hidden_dim=cfg.get("hidden_dim", 512), **common)
    if name == "cnn":
        return CNNClassifier(hidden_dim=cfg.get("hidden_dim", 256), **common)
    transformer = {
        **common,
        "patch_size": cfg.get("patch_size", 4),
        "embed_dim": cfg.get("embed_dim", 128),
        "depth": cfg.get("depth", 4),
        "num_heads": cfg.get("num_heads", 4),
        "mlp_ratio": cfg.get("mlp_ratio", 4.0),
        "pooling": cfg.get("pooling", "cls"),
    }
    if name == "vit":
        return TinyViT(**transformer)
    if name == "hybrid_vit":
        return HybridConvViT(**transformer)
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "MLPClassifier",
    "CNNClassifier",
    "PatchEmbedding",
    "TinyViT",
    "HybridConvViT",
    "build_model",
]

