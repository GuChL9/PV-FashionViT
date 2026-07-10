#!/usr/bin/env bash
set -euo pipefail
for config in configs/mlp.yaml configs/cnn.yaml configs/vit_abspos.yaml configs/vit_aug.yaml configs/vit_meanpool.yaml configs/hybrid_vit.yaml; do
  python src/main.py --config "$config"
done

