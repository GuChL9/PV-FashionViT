import pytest
import torch

from src.models import CNNClassifier, HybridConvViT, MLPClassifier, TinyViT
from src.utils.attention import attention_rollout, forward_with_attention


@pytest.mark.parametrize(
    "model",
    [
        MLPClassifier(hidden_dim=32),
        CNNClassifier(hidden_dim=32),
        TinyViT(embed_dim=32, depth=1, num_heads=4, mlp_ratio=2, pooling="cls"),
        TinyViT(embed_dim=32, depth=1, num_heads=4, mlp_ratio=2, pooling="mean"),
        HybridConvViT(embed_dim=32, depth=1, num_heads=4, mlp_ratio=2, pooling="mean"),
    ],
)
def test_model_forward_shape(model):
    output = model(torch.randn(2, 1, 56, 56))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("encoding", ["learnable", "sincos", "none"])
def test_vit_positional_encoding_variants(encoding):
    model = TinyViT(embed_dim=32, depth=1, num_heads=4, mlp_ratio=2, positional_encoding=encoding)
    output = model(torch.randn(2, 1, 56, 56))
    assert output.shape == (2, 10)


def test_attention_rollout_shape():
    model = TinyViT(embed_dim=32, depth=2, num_heads=4, mlp_ratio=2, pooling="cls").eval()
    logits, layers = forward_with_attention(model, torch.randn(2, 1, 56, 56))
    maps = attention_rollout(layers, model.pooling)
    assert logits.shape == (2, 10)
    assert len(layers) == 2
    assert maps.shape == (2, 14, 14)
    assert torch.isfinite(maps).all()
