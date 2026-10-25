#!/usr/bin/env bash
# J1 (E17-E18): unit tests, sampler audit + fixed unseen pool, then the 2x2 training runs.
# Requires the phase-3 split under ../dataset/clinc_phase3 (phase3_scl/04_steps/01_prepare_split.py).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
start_log phase4_train

"$PYTHON_BIN" -m pytest 06_tests phase3_scl/06_tests phase4_dascl/06_tests -q
"$PYTHON_BIN" phase4_dascl/04_steps/01_audit_and_fix_unseen.py

C=phase4_dascl/01_configs
for seed in 0 1 2 3 4 5 6 7 8 9; do            # primary: 10-shot, 10 seeds
  for arm in ib_ce ib_scl da_ce da_scl; do
    train "$C/arm_n10_${arm}.yaml" "$seed"
  done
done
for seed in 0 1 2 3 4; do                      # descriptive replication: 25-shot, 5 seeds
  for arm in ib_ce ib_scl da_ce da_scl; do
    train "$C/arm_n25_${arm}.yaml" "$seed"
  done
done
nvidia-smi || true
