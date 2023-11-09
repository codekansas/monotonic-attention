# Makefile

define HELP_MESSAGE
                    Monotonic Attention
                    -------------------

Welcome to the `monotonic-attention` package!

# Installing

1. Create a new Conda environment: `conda create --name monotonic-attention python=3.10`
2. Activate the environment: `conda activate monotonic-attention`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#          Build           #
# ------------------------ #

install-torch-nightly:
	@pip install --pre torch --index-url https://download.pytorch.org/whl/nightly
.PHONY: install-torch-nightly

install:
	@pip install --verbose -e .
.PHONY: install

install-dev:
	@pip install --verbose -e '.[dev]'
.PHONY: install

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #

py-files := $(filter-out ml/api.py, $(shell git ls-files '*.py'))

format:
	@black $(py-files)
	@ruff --fix $(py-files)
.PHONY: format

static-checks:
	@black --diff --check $(py-files)
	@ruff $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint

mypy-daemon:
	@dmypy run -- $(py-files)
.PHONY: mypy-daemon

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
