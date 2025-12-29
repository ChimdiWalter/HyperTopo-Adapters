#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/default.yaml}
python -m src.train --config ${CFG} --dry_run
python -m src.train --config ${CFG}
python -m src.eval  --config ${CFG}
