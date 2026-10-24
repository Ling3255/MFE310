#!/usr/bin/env bash
# J02: Verify explicit approval and evaluate the frozen artifact set once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/03_src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FROZEN_CONFIG="${FROZEN_CONFIG:-01_configs/04_frozen.yaml}"

"$PYTHON_BIN" - "$FROZEN_CONFIG" <<'PY'
import sys
import yaml
from datetime import datetime

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
approval = config.get("supervisor_approval", {})
if not config.get("frozen") or not approval.get("approved"):
    raise SystemExit("E6 blocked: frozen supervisor approval is not recorded")
for key in ("confirmed_by", "confirmed_at", "evidence"):
    if not approval.get(key):
        raise SystemExit(f"E6 blocked: supervisor_approval.{key} is missing")
datetime.fromisoformat(str(approval["confirmed_at"]).replace("Z", "+00:00"))
PY

mkdir -p 09_results/job_logs
LOG="09_results/job_logs/evaluate_frozen_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
"$PYTHON_BIN" 04_steps/06_evaluate_test.py --config "$FROZEN_CONFIG"
