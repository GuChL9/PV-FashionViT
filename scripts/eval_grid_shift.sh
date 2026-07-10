#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/vit_abspos.yaml}"
CHECKPOINT="${2:-}"
if [[ -n "$CHECKPOINT" ]]; then
  python src/main.py --config "$CONFIG" --eval-only --grid --checkpoint "$CHECKPOINT"
else
  python src/main.py --config "$CONFIG" --eval-only --grid
fi
