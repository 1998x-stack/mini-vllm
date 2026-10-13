.PHONY: test examples check example-v07 example-v08 example-v09

test:
	python -m pytest

examples:
	python examples/01_basic_generate.py
	python examples/02_static_vs_continuous.py
	python examples/03_page_table_walkthrough.py
	python examples/04_compare_cache_waste.py
	python examples/05_chunked_prefill.py
	python examples/06_prefix_caching.py
	python examples/07_preemption.py
	python examples/08_speculative_decoding.py
	python examples/09_executor_boundary.py
	python examples/10_tensor_parallel.py

check:
	./scripts/check.sh

example-v07:
	PYTHONPATH=. python examples/08_speculative_decoding.py

example-v08:
	PYTHONPATH=. python examples/09_executor_boundary.py

example-v09:
	PYTHONPATH=. python examples/10_tensor_parallel.py
