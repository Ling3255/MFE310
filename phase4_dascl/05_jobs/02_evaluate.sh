#!/usr/bin/env bash
# J2 (E19): run ONCE after the supervisor approved phase4_dascl/01_configs/study.yaml.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
start_log phase4_evaluate

"$PYTHON_BIN" phase4_dascl/04_steps/02_evaluate_test.py
"$PYTHON_BIN" phase4_dascl/04_steps/03_statistics.py
"$PYTHON_BIN" phase4_dascl/04_steps/04_analyze.py
