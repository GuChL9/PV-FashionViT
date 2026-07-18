param(
    [int[]]$Seeds = @(42, 2026, 3407),
    [switch]$Force,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$profiles = @(
    @{ Config = "vit_center_stage"; RunBase = "vit_center_stage_cpu" },
    @{ Config = "vit_sincos_center_cls"; RunBase = "vit_sincos_center_cls_cpu" },
    @{ Config = "vit_no_pos_center_cls"; RunBase = "vit_no_pos_center_cls_cpu" }
)

foreach ($profile in $profiles) {
    foreach ($seed in $Seeds) {
        $config = "configs/ablations/$($profile.Config).yaml"
        $runName = "$($profile.RunBase)_s${seed}"
        $evaluation = "outputs/${runName}/evaluation.json"
        if ((-not $Force) -and (Test-Path $evaluation)) {
            Write-Host "Skipping completed run: $runName"
            continue
        }
        & $Python src/main.py --config $config --seed $seed --run-name $runName
        if ($LASTEXITCODE -ne 0) {
            throw "Position-encoding training failed for $runName"
        }
    }
}

& $Python src/analyze_position_encoding_ablation.py --seeds $Seeds
if ($LASTEXITCODE -ne 0) {
    throw "Failed to summarize the position-encoding ablation"
}
