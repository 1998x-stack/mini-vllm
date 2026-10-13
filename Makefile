.PHONY: test examples check example-v07 example-v08 example-v09 example-v10 example-v11 example-v12

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
	python examples/11_async_streaming.py
	python examples/12_compile_aware.py
	python examples/13_paged_attention.py

check:
	./scripts/check.sh

example-v07:
	PYTHONPATH=. python examples/08_speculative_decoding.py

example-v08:
	PYTHONPATH=. python examples/09_executor_boundary.py

example-v09:
	PYTHONPATH=. python examples/10_tensor_parallel.py

example-v10:
	PYTHONPATH=. python examples/11_async_streaming.py

example-v11:
	PYTHONPATH=. python examples/12_compile_aware.py

example-v12:
	PYTHONPATH=. python examples/13_paged_attention.py
