.PHONY: test examples check

test:
	python -m pytest

examples:
	python examples/01_basic_generate.py
	python examples/02_static_vs_continuous.py

check:
	./scripts/check.sh
