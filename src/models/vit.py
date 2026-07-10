from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 56, patch_size: int = 4, in_channels: int = 1,
                 embed_dim: int = 128) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyViT(nn.Module):
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
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.pooling = pooling
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        token_count = self.patch_embed.num_patches + (pooling == "cls")
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder(x + self.pos_embed)
        x = self.norm(x)
        return x[:, 0] if self.pooling == "cls" else x.mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))
