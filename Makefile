.PHONY: test examples check

test:
	python -m pytest

examples:
	python examples/01_basic_generate.py
	python examples/02_static_vs_continuous.py
	python examples/03_page_table_walkthrough.py
	python examples/04_compare_cache_waste.py
	python examples/05_chunked_prefill.py

check:
	./scripts/check.sh
