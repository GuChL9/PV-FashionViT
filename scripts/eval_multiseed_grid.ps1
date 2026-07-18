$ErrorActionPreference = "Stop"

$models = @(
    @{ ConfigStem = "mlp"; RunBase = "mlp_center_cpu" },
    @{ ConfigStem = "cnn"; RunBase = "cnn_center_cpu" },
    @{ ConfigStem = "vit_abspos"; RunBase = "vit_abspos_center_cpu" },
    @{ ConfigStem = "vit_aug"; RunBase = "vit_aug_cpu" },
    @{ ConfigStem = "vit_meanpool"; RunBase = "vit_meanpool_cpu" },
    @{ ConfigStem = "hybrid_vit"; RunBase = "hybrid_vit_cpu" }
)
$seeds = @(42, 2026, 3407, 12345, 98765)

foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $config = "configs/seeds/$($model.ConfigStem)_s${seed}.yaml"
        $checkpoint = "outputs/$($model.RunBase)_s${seed}/checkpoints/best.pt"
        if (-not (Test-Path $checkpoint)) {
            throw "Missing checkpoint: $checkpoint. Run scripts/run_multiseed_experiments.ps1 first."
        }
        python src/main.py --config $config --eval-only --grid --checkpoint $checkpoint
    }
}

# Seed 42 supplies the detailed per-run figures; all five seeds supply the
# aggregate tables and uncertainty plots. Both are rebuilt from the same 30 runs.
python src/analyze_results.py
python src/analyze_multiseed_results.py
