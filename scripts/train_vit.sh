#!/usr/bin/env bash
set -euo pipefail
python src/main.py --config configs/vit_abspos.yaml
python src/main.py --config configs/vit_aug.yaml
python src/main.py --config configs/vit_meanpool.yaml

