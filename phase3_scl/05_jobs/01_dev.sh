#!/usr/bin/env bash
# J1 (E12): fixed split, B0 baseline, development runs on seeds 100-102, then validation-only selection.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
start_log phase3_dev

"$PYTHON_BIN" phase3_scl/04_steps/01_prepare_split.py
"$PYTHON_BIN" 04_steps/02_train_baseline.py --config phase3_scl/01_configs/baseline.yaml
for seed in 100 101 102; do
  train phase3_scl/01_configs/arm_ce_balanced.yaml "$seed"
  for config in phase3_scl/01_configs/dev/arm_scl_*.yaml; do
    train "$config" "$seed"
  done
done
"$PYTHON_BIN" phase3_scl/04_steps/02_select_scl.py
nvidia-smi || true
