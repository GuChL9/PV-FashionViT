$ErrorActionPreference = "Stop"
$configs = @(
    "configs/mlp.yaml",
    "configs/cnn.yaml",
    "configs/vit_abspos.yaml",
    "configs/vit_aug.yaml",
    "configs/vit_meanpool.yaml",
    "configs/hybrid_vit.yaml"
)
foreach ($config in $configs) {
    python src/main.py --config $config
}

