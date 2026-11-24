#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/run_experiments.py --configs \
  configs/experiments/e0_baseline.yaml \
  configs/experiments/e2_sobel.yaml
