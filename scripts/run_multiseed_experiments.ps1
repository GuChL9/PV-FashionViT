$ErrorActionPreference = "Stop"

$configs = @(
    "configs/seeds/mlp_s42.yaml",
    "configs/seeds/mlp_s2026.yaml",
    "configs/seeds/mlp_s3407.yaml",
    "configs/seeds/mlp_s12345.yaml",
    "configs/seeds/mlp_s98765.yaml",
    "configs/seeds/cnn_s42.yaml",
    "configs/seeds/cnn_s2026.yaml",
    "configs/seeds/cnn_s3407.yaml",
    "configs/seeds/cnn_s12345.yaml",
    "configs/seeds/cnn_s98765.yaml",
    "configs/seeds/vit_abspos_s42.yaml",
    "configs/seeds/vit_abspos_s2026.yaml",
    "configs/seeds/vit_abspos_s3407.yaml",
    "configs/seeds/vit_abspos_s12345.yaml",
    "configs/seeds/vit_abspos_s98765.yaml",
    "configs/seeds/vit_aug_s42.yaml",
    "configs/seeds/vit_aug_s2026.yaml",
    "configs/seeds/vit_aug_s3407.yaml",
    "configs/seeds/vit_aug_s12345.yaml",
    "configs/seeds/vit_aug_s98765.yaml",
    "configs/seeds/vit_meanpool_s42.yaml",
    "configs/seeds/vit_meanpool_s2026.yaml",
    "configs/seeds/vit_meanpool_s3407.yaml",
    "configs/seeds/vit_meanpool_s12345.yaml",
    "configs/seeds/vit_meanpool_s98765.yaml",
    "configs/seeds/hybrid_vit_s42.yaml",
    "configs/seeds/hybrid_vit_s2026.yaml",
    "configs/seeds/hybrid_vit_s3407.yaml",
    "configs/seeds/hybrid_vit_s12345.yaml",
    "configs/seeds/hybrid_vit_s98765.yaml"
)

foreach ($config in $configs) {
    python src/main.py --config $config
}
