#!/usr/bin/env bash
set -euo pipefail
python src/main.py --config configs/mlp.yaml
python src/main.py --config configs/cnn.yaml
