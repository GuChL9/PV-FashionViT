param(
    [int]$Seed = 42,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$profiles = @(
    @{
        Config = "configs/ablations/vit_center_stage.yaml"
        RunName = "vit_center_stage_cpu_s${Seed}"
    },
    @{
        Config = "configs/ablations/vit_shift_rotation_erasing.yaml"
        RunName = "vit_shift_rotation_erasing_cpu_s${Seed}"
    }
)

foreach ($profile in $profiles) {
    $checkpoint = "outputs/$($profile.RunName)/checkpoints/best.pt"
    if (-not (Test-Path $checkpoint)) {
        throw "Missing checkpoint: $checkpoint"
    }
    & $Python src/evaluate_matched_clipping.py `
        --config $profile.Config `
        --checkpoint $checkpoint `
        --seed $Seed `
        --run-name $profile.RunName
    if ($LASTEXITCODE -ne 0) {
        throw "Matched clipping evaluation failed for $($profile.RunName)"
    }
}

& $Python src/analyze_matched_clipping.py
if ($LASTEXITCODE -ne 0) {
    throw "Failed to analyze matched clipping results"
}
