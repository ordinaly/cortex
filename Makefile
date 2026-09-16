.PHONY: test native-test fmt clippy golden

test:
	pytest tests/test_freeze.py

native-test:
	cargo test -p cortex-core --release
	cargo check -p cortex-python --release
	python -m pip install -e .
	pytest tests/test_api.py tests/test_golden.py

fmt:
	cargo fmt --all

clippy:
	cargo clippy -p cortex-core --all-targets -- -D warnings

golden:
	python scripts/generate_golden.py
