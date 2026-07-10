import pytest
import torch

from src.models import CNNClassifier, HybridConvViT, MLPClassifier, TinyViT


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

