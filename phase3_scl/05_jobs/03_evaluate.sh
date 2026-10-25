#!/usr/bin/env bash
# J3 (E15/E16): run ONCE after the supervisor approved study.yaml. The test step refuses a second run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
start_log phase3_evaluate

"$PYTHON_BIN" phase3_scl/04_steps/03_evaluate_test.py
"$PYTHON_BIN" phase3_scl/04_steps/04_statistics.py
"$PYTHON_BIN" phase3_scl/04_steps/05_analyze.py
