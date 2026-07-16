$ErrorActionPreference = "Stop"

$models = @("mlp", "cnn", "vit_abspos", "vit_aug", "vit_meanpool", "hybrid_vit")
$seeds = @(42, 2026, 3407, 12345, 98765)

foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $config = "configs/seeds/${model}_s${seed}.yaml"
        python src/main.py --config $config
    }
}
