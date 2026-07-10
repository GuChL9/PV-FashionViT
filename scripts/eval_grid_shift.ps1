param(
    [string]$Config = "configs/vit_abspos.yaml",
    [string]$Checkpoint = "outputs/vit_abspos_center_cpu/checkpoints/best.pt"
)

$ErrorActionPreference = "Stop"
python src/main.py --config $Config --eval-only --grid --checkpoint $Checkpoint
