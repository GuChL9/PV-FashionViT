param(
    [string]$Config = "configs/seeds/vit_abspos_s42.yaml",
    [string]$Checkpoint = "outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt"
)

$ErrorActionPreference = "Stop"
python src/main.py --config $Config --eval-only --grid --checkpoint $Checkpoint
