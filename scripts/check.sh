#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python -m pytest
python examples/01_basic_generate.py >/dev/null
python examples/02_static_vs_continuous.py >/dev/null
python examples/03_page_table_walkthrough.py >/dev/null
python examples/04_compare_cache_waste.py >/dev/null
python examples/05_chunked_prefill.py >/dev/null
python examples/06_prefix_caching.py >/dev/null
python examples/07_preemption.py >/dev/null
python examples/08_speculative_decoding.py >/dev/null
python examples/09_executor_boundary.py >/dev/null
python examples/10_tensor_parallel.py >/dev/null
python examples/11_async_streaming.py >/dev/null
python examples/12_compile_aware.py >/dev/null
python examples/13_paged_attention.py >/dev/null
python examples/14_prefix_aware_scheduling.py >/dev/null
python examples/15_integrated_runtime.py >/dev/null
python examples/16_multiprocess_transport.py >/dev/null
python examples/17_model_loading.py >/dev/null
python examples/18_network_server.py >/dev/null
echo "all checks passed"
