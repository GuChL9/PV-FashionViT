$ErrorActionPreference = "Stop"

$runs = @(
    @{ Config = "configs/seeds/mlp_s42.yaml"; Checkpoint = "outputs/mlp_center_cpu_s42/checkpoints/best.pt" },
    @{ Config = "configs/seeds/mlp_s2026.yaml"; Checkpoint = "outputs/mlp_center_cpu_s2026/checkpoints/best.pt" },
    @{ Config = "configs/seeds/mlp_s3407.yaml"; Checkpoint = "outputs/mlp_center_cpu_s3407/checkpoints/best.pt" },
    @{ Config = "configs/seeds/mlp_s12345.yaml"; Checkpoint = "outputs/mlp_center_cpu_s12345/checkpoints/best.pt" },
    @{ Config = "configs/seeds/mlp_s98765.yaml"; Checkpoint = "outputs/mlp_center_cpu_s98765/checkpoints/best.pt" },
    @{ Config = "configs/seeds/cnn_s42.yaml"; Checkpoint = "outputs/cnn_center_cpu_s42/checkpoints/best.pt" },
    @{ Config = "configs/seeds/cnn_s2026.yaml"; Checkpoint = "outputs/cnn_center_cpu_s2026/checkpoints/best.pt" },
    @{ Config = "configs/seeds/cnn_s3407.yaml"; Checkpoint = "outputs/cnn_center_cpu_s3407/checkpoints/best.pt" },
    @{ Config = "configs/seeds/cnn_s12345.yaml"; Checkpoint = "outputs/cnn_center_cpu_s12345/checkpoints/best.pt" },
    @{ Config = "configs/seeds/cnn_s98765.yaml"; Checkpoint = "outputs/cnn_center_cpu_s98765/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_abspos_s42.yaml"; Checkpoint = "outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_abspos_s2026.yaml"; Checkpoint = "outputs/vit_abspos_center_cpu_s2026/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_abspos_s3407.yaml"; Checkpoint = "outputs/vit_abspos_center_cpu_s3407/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_abspos_s12345.yaml"; Checkpoint = "outputs/vit_abspos_center_cpu_s12345/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_abspos_s98765.yaml"; Checkpoint = "outputs/vit_abspos_center_cpu_s98765/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_aug_s42.yaml"; Checkpoint = "outputs/vit_aug_cpu_s42/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_aug_s2026.yaml"; Checkpoint = "outputs/vit_aug_cpu_s2026/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_aug_s3407.yaml"; Checkpoint = "outputs/vit_aug_cpu_s3407/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_aug_s12345.yaml"; Checkpoint = "outputs/vit_aug_cpu_s12345/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_aug_s98765.yaml"; Checkpoint = "outputs/vit_aug_cpu_s98765/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_meanpool_s42.yaml"; Checkpoint = "outputs/vit_meanpool_cpu_s42/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_meanpool_s2026.yaml"; Checkpoint = "outputs/vit_meanpool_cpu_s2026/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_meanpool_s3407.yaml"; Checkpoint = "outputs/vit_meanpool_cpu_s3407/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_meanpool_s12345.yaml"; Checkpoint = "outputs/vit_meanpool_cpu_s12345/checkpoints/best.pt" },
    @{ Config = "configs/seeds/vit_meanpool_s98765.yaml"; Checkpoint = "outputs/vit_meanpool_cpu_s98765/checkpoints/best.pt" },
    @{ Config = "configs/seeds/hybrid_vit_s42.yaml"; Checkpoint = "outputs/hybrid_vit_cpu_s42/checkpoints/best.pt" },
    @{ Config = "configs/seeds/hybrid_vit_s2026.yaml"; Checkpoint = "outputs/hybrid_vit_cpu_s2026/checkpoints/best.pt" },
    @{ Config = "configs/seeds/hybrid_vit_s3407.yaml"; Checkpoint = "outputs/hybrid_vit_cpu_s3407/checkpoints/best.pt" },
    @{ Config = "configs/seeds/hybrid_vit_s12345.yaml"; Checkpoint = "outputs/hybrid_vit_cpu_s12345/checkpoints/best.pt" },
    @{ Config = "configs/seeds/hybrid_vit_s98765.yaml"; Checkpoint = "outputs/hybrid_vit_cpu_s98765/checkpoints/best.pt" }
)

foreach ($run in $runs) {
    if (-not (Test-Path $run.Checkpoint)) {
        throw "Missing checkpoint: $($run.Checkpoint). Run scripts/run_multiseed_experiments.ps1 first."
    }
    python src/main.py --config $run.Config --eval-only --grid --checkpoint $run.Checkpoint
}
