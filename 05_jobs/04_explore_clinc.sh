#!/usr/bin/env bash
# J04 (optional, E11): exploratory CLINC variants (smaller lambda, 10 epochs); validation only, never test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/03_src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p 08_runs/job_logs
LOG="08_runs/job_logs/explore_clinc_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

for seed in 0 1 2; do
  for lambda in 0 0.02 0.05 0.1; do
    "$PYTHON_BIN" 04_steps/03_train_bert.py --config 01_configs/08_explore_clinc_base.yaml --lambda-domain "$lambda" --seed "$seed"
  done
done
"$PYTHON_BIN" 04_steps/12_confirmatory_statistics.py --config 01_configs/09_explore_clinc.yaml --split val
