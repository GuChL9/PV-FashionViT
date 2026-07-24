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
    assert config["data"]["rotation_degrees"] == 45
    assert config["data"]["angle_values"] == [-45, -30, -15, 0, 15, 30, 45]


def test_gpu_profile_keeps_full_model():
    config = load_config("configs/gpu/vit_abspos.yaml")
    assert config["model"]["patch_size"] == 4
    assert config["model"]["embed_dim"] == 128
    assert config["model"]["depth"] == 4
    assert config["train"]["epochs"] == 30


def test_sincos_profile_is_explicit():
    config = load_config("configs/vit_sincos.yaml")
    assert config["model"]["positional_encoding"] == "sincos"
    assert config["model"]["pooling"] == "mean"


def test_augmentation_ablation_changes_one_training_component_at_a_time():
    shift = load_config("configs/ablations/vit_shift.yaml")
    shift_rotation = load_config("configs/ablations/vit_shift_rotation.yaml")
    assert shift["data"]["train_mode"] == "random_shift"
    assert shift["data"]["random_erasing_prob"] == 0.0
    assert shift_rotation["data"]["train_mode"] == "shift_rotation"
    assert shift_rotation["data"]["random_erasing_prob"] == 0.0
    assert shift["model"] == shift_rotation["model"]
    assert shift["output"]["publish_global"] is False


def test_position_encoding_ablation_is_controlled():
    learnable = load_config("configs/ablations/vit_center_stage.yaml")
    sincos = load_config("configs/ablations/vit_sincos_center_cls.yaml")
    no_pos = load_config("configs/ablations/vit_no_pos_center_cls.yaml")
    assert learnable["data"]["train_mode"] == "center"
    assert {profile["model"]["pooling"] for profile in (learnable, sincos, no_pos)} == {"cls"}
    assert [
        profile["model"]["positional_encoding"]
        for profile in (learnable, sincos, no_pos)
    ] == ["learnable", "sincos", "none"]
    for profile in (learnable, sincos, no_pos):
        assert profile["output"]["publish_global"] is False
