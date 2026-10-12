.PHONY: test examples check example-v07

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

check:
	./scripts/check.sh

example-v07:
	PYTHONPATH=. python examples/08_speculative_decoding.py
