from src.utils.config import load_config


def test_config_inheritance():
    config = load_config("configs/vit_meanpool.yaml")
    assert config["model"]["name"] == "vit"
    assert config["model"]["pooling"] == "mean"
    assert config["data"]["canvas_size"] == 56
    assert config["device"] == "cpu"
    assert config["model"]["patch_size"] == 7
    assert config["model"]["embed_dim"] == 64
    assert config["data"]["num_workers"] == 0


def test_gpu_profile_keeps_full_model():
    config = load_config("configs/gpu/vit_abspos.yaml")
    assert config["model"]["patch_size"] == 4
    assert config["model"]["embed_dim"] == 128
    assert config["model"]["depth"] == 4
    assert config["train"]["epochs"] == 30
