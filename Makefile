.PHONY: test examples check

test:
	python -m pytest

examples:
	python examples/01_basic_generate.py

check:
	./scripts/check.sh
