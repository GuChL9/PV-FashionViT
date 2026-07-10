#!/usr/bin/env bash
set -euo pipefail

declare -a runs=(
  "configs/mlp.yaml outputs/mlp_center_cpu/checkpoints/best.pt"
  "configs/cnn.yaml outputs/cnn_center_cpu/checkpoints/best.pt"
  "configs/vit_abspos.yaml outputs/vit_abspos_center_cpu/checkpoints/best.pt"
  "configs/vit_aug.yaml outputs/vit_aug_cpu/checkpoints/best.pt"
  "configs/vit_meanpool.yaml outputs/vit_meanpool_cpu/checkpoints/best.pt"
  "configs/hybrid_vit.yaml outputs/hybrid_vit_cpu/checkpoints/best.pt"
)

for run in "${runs[@]}"; do
  read -r config checkpoint <<< "$run"
  python src/main.py --config "$config" --eval-only --grid --checkpoint "$checkpoint"
done

python src/analyze_results.py
