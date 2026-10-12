#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python -m pytest
python examples/01_basic_generate.py >/dev/null
python examples/02_static_vs_continuous.py >/dev/null
python examples/03_page_table_walkthrough.py >/dev/null
python examples/04_compare_cache_waste.py >/dev/null
echo "all checks passed"
