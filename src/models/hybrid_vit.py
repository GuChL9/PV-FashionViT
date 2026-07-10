from __future__ import annotations

import torch
from torch import nn

from .vit import build_2d_sincos_position_embedding


class HybridConvViT(nn.Module):
    def __init__(
        self,
        img_size: int = 56,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pooling: str = "mean",
        positional_encoding: str = "learnable",
    ) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        if positional_encoding not in {"learnable", "sincos", "none"}:
            raise ValueError("positional_encoding must be 'learnable', 'sincos', or 'none'")
        self.pooling = pooling
        self.positional_encoding = positional_encoding
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.patch_embed = nn.Conv2d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        token_count = num_patches + (pooling == "cls")
        if positional_encoding == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif positional_encoding == "sincos":
            self.register_buffer(
                "pos_embed",
                build_2d_sincos_position_embedding(img_size // patch_size, embed_dim, pooling == "cls"),
            )
        else:
            self.register_buffer("pos_embed", torch.zeros(1, token_count, embed_dim), persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pooling == "cls" else None
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(self.stem(x)).flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.norm(self.encoder(x + self.pos_embed))
        pooled = x[:, 0] if self.pooling == "cls" else x.mean(dim=1)
        return self.head(pooled)
