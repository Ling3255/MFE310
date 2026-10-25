#!/usr/bin/env bash
# J2 (E14 + R2): confirmatory runs. Requires phase3_scl/01_configs/generated/ from 01_dev.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
start_log phase3_confirm

GEN=phase3_scl/01_configs/generated
for config in "$GEN/arm_scl_selected.yaml" "$GEN/arm_scl_selected_n10.yaml" "$GEN/arm_scl_selected_n25.yaml"; do
  [[ -f "$config" ]] || { echo "Missing $config: run 01_dev.sh first (or no SCL candidate was eligible)"; exit 1; }
done
for seed in 0 1 2 3 4 5 6 7 8 9; do
  train phase3_scl/01_configs/arm_ce_balanced.yaml "$seed"   # A  principal model
  train "$GEN/arm_scl_selected.yaml" "$seed"                 # B  controlled improvement
  train phase3_scl/01_configs/arm_ce_standard.yaml "$seed"   # A' sampler ablation
done
for seed in 0 1 2 3 4; do                                    # R2 low-resource
  for size in 10 25; do
    train "phase3_scl/01_configs/arm_ce_balanced_n${size}.yaml" "$seed"
    train "$GEN/arm_scl_selected_n${size}.yaml" "$seed"
  done
done
nvidia-smi || true
