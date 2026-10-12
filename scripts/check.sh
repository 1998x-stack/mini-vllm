#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python -m pytest
python examples/01_basic_generate.py >/dev/null
echo "all checks passed"
