$ErrorActionPreference = "Stop"

$runs = @(
    @{ Config = "configs/mlp.yaml"; Checkpoint = "outputs/mlp_center_cpu/checkpoints/best.pt" },
    @{ Config = "configs/cnn.yaml"; Checkpoint = "outputs/cnn_center_cpu/checkpoints/best.pt" },
    @{ Config = "configs/vit_abspos.yaml"; Checkpoint = "outputs/vit_abspos_center_cpu/checkpoints/best.pt" },
    @{ Config = "configs/vit_aug.yaml"; Checkpoint = "outputs/vit_aug_cpu/checkpoints/best.pt" },
    @{ Config = "configs/vit_meanpool.yaml"; Checkpoint = "outputs/vit_meanpool_cpu/checkpoints/best.pt" },
    @{ Config = "configs/hybrid_vit.yaml"; Checkpoint = "outputs/hybrid_vit_cpu/checkpoints/best.pt" }
)

foreach ($run in $runs) {
    python src/main.py --config $run.Config --eval-only --grid --checkpoint $run.Checkpoint
}

python src/analyze_results.py
