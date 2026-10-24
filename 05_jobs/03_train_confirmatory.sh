#!/usr/bin/env bash
# J03: Train the confirmatory runs (E9 CLINC seeds 3-9, E10 MASSIVE seeds 0-9), then validation sanity checks.
# Usage: bash 05_jobs/03_train_confirmatory.sh [clinc|massive|all]
# Do not pass --data-root: the data path is part of config_hash and S11 verifies runs used the base config unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/03_src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-all}"
mkdir -p 08_runs/job_logs
LOG="08_runs/job_logs/train_confirmatory_${TARGET}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

train() {  # config lambda seed
  "$PYTHON_BIN" 04_steps/03_train_bert.py --config "$1" --lambda-domain "$2" --seed "$3"
}

if [[ "$TARGET" == clinc || "$TARGET" == all ]]; then
  for seed in 3 4 5 6 7 8 9; do
    for lambda in 0 0.1; do train 01_configs/01_base.yaml "$lambda" "$seed"; done
  done
  "$PYTHON_BIN" 04_steps/12_confirmatory_statistics.py --config 01_configs/05_confirm_clinc.yaml --split val
fi

if [[ "$TARGET" == massive || "$TARGET" == all ]]; then
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    for lambda in 0 0.1; do train 01_configs/07_massive_base.yaml "$lambda" "$seed"; done
  done
  "$PYTHON_BIN" 04_steps/12_confirmatory_statistics.py --config 01_configs/06_confirm_massive.yaml --split val
fi
