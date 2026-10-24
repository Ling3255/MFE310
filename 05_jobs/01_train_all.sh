#!/usr/bin/env bash
# J01: Run the 5 lambda x 3 seed formal training matrix serially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing data_full.json and domains.json}"
export PYTHONPATH="$ROOT/03_src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p 08_runs/job_logs
LOG="08_runs/job_logs/train_all_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

for lambda in 0 0.1 0.3 0.5 1.0; do
  for seed in 0 1 2; do
    "$PYTHON_BIN" 04_steps/03_train_bert.py \
      --config 01_configs/01_base.yaml \
      --data-root "$DATA_ROOT" \
      --lambda-domain "$lambda" \
      --seed "$seed"
  done
done

"$PYTHON_BIN" 04_steps/04_select_lambda.py --config 01_configs/03_search.yaml
