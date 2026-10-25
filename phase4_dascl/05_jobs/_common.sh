#!/usr/bin/env bash
# Shared setup for phase-4 jobs: repository root, PYTHONPATH, log file and environment evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/03_src:$ROOT/phase3_scl/03_src:$ROOT/phase4_dascl/03_src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p phase4_dascl/08_runs/job_logs

start_log() {  # start_log <job-name>: tee everything into a timestamped log and record the environment
  LOG="phase4_dascl/08_runs/job_logs/$1_$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$LOG") 2>&1
  echo "== job $1 started $(date -u +%FT%TZ) on $(uname -a 2>/dev/null || echo unknown)"
  "$PYTHON_BIN" -c "import json, clinc_domain_aux.common as c; print(json.dumps(c.environment_info(), indent=1))"
  nvidia-smi || echo "nvidia-smi not available"
}

train() {  # train <config> <seed>: shared S03 entry point (never pass --data-root)
  "$PYTHON_BIN" 04_steps/03_train_bert.py --config "$1" --lambda-domain 0 --seed "$2"
}
