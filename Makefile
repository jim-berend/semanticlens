.PHONY: install
install:
	pip install -e .

.PHONY: install-dev
install-dev:
	pip install -e .[dev]
	pre-commit install

# .PHONY: type
# type:
# 	SKIP=no-commit-to-branch pre-commit run -a pyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	SKIP=no-commit-to-branch pre-commit run -a ruff-lint
	SKIP=no-commit-to-branch pre-commit run -a ruff-format

.PHONY: check
check:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

.PHONY: test
test:
	python -m pytest tests/

.PHONY: test-cov
test-cov:
	python -m pytest tests/ --cov=semanticlens --cov-report=term-missing --cov-fail-under=85

