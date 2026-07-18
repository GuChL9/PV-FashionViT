param(
    [int[]]$Seeds = @(42, 2026, 3407),
    [switch]$Grid,
    [switch]$Force,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$stages = @(
    @{ Config = "vit_center_stage"; RunBase = "vit_center_stage_cpu" },
    @{ Config = "vit_shift"; RunBase = "vit_shift_cpu" },
    @{ Config = "vit_shift_rotation"; RunBase = "vit_shift_rotation_cpu" },
    @{ Config = "vit_shift_rotation_erasing"; RunBase = "vit_shift_rotation_erasing_cpu" }
)

foreach ($stage in $stages) {
    foreach ($seed in $Seeds) {
        $config = "configs/ablations/$($stage.Config).yaml"
        $runName = "$($stage.RunBase)_s${seed}"
        if (-not (Test-Path $config)) {
            throw "Missing ablation config: $config"
        }
        $evaluation = "outputs/${runName}/evaluation.json"
        if ((-not $Force) -and (Test-Path $evaluation)) {
            Write-Host "Skipping completed run: $runName"
            continue
        }
        $arguments = @(
            "src/main.py", "--config", $config,
            "--seed", $seed, "--run-name", $runName
        )
        if ($Grid) {
            $arguments += "--grid"
        }
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation training failed for $config"
        }
    }
}

& $Python src/analyze_augmentation_ablation.py --seeds $Seeds
if ($LASTEXITCODE -ne 0) {
    throw "Failed to summarize the augmentation ablation"
}
