param(
    [int[]]$Seeds = @(42),
    [switch]$Grid
)

$ErrorActionPreference = "Stop"
$stages = @("vit_shift", "vit_shift_rotation")

foreach ($stage in $stages) {
    foreach ($seed in $Seeds) {
        $config = "configs/ablations/${stage}_s${seed}.yaml"
        if (-not (Test-Path $config)) {
            throw "Missing ablation config: $config"
        }
        $arguments = @("src/main.py", "--config", $config)
        if ($Grid) {
            $arguments += "--grid"
        }
        python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation training failed for $config"
        }
    }
}

if ($Seeds -contains 42) {
    python src/analyze_augmentation_ablation.py --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to summarize seed 42 augmentation ablation"
    }
}
