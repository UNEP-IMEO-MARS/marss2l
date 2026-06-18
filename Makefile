.PHONY: conda black isort format ruff-check-all ruff-check-missing-imports lint test build publish mount-container help condaenv check-condaenv lock
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.12
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = marss2l
NOTEBOOK_KERNEL ?= python3

# All action targets run inside this conda env; create it with `make condaenv`.
# We call the env's binaries by absolute path (not `conda run -n ...`) so the right
# env is targeted even when another conda env is already activated in the shell.
CONDA_ENV = marss2lpy312_dev
CONDA_BASE := $(shell $(CONDA) info --base)
ENV_BIN = $(CONDA_BASE)/envs/$(CONDA_ENV)/bin
LOCKFILE = environment/requirements-test.lock

# Notebooks exercised as integration tests.
NOTEBOOKS = \
	notebooks/examples/download_and_inference.ipynb \
	notebooks/examples/background_image_selection.ipynb \
	notebooks/examples/plot_images_dataset_train.ipynb \
	notebooks/examples/plot_plumes_dataset_test.ipynb \
	notebooks/examples/run_inference.ipynb \
	notebooks/figures/dataset_stats_by_split_and_geopackage_locations.ipynb \
	notebooks/figures/figure_number_of_images_per_country.ipynb \
	notebooks/figures/mdl_exploration_by_case_study.ipynb \
	notebooks/figures/mdl_exploration_adapted.ipynb \
	notebooks/figures/figure_wind_speed.ipynb \
	notebooks/figures/stats_dataset_toareflectances.ipynb \
	notebooks/figures/eval_model_and_figure_prob_vs_emission_rate.ipynb \
	notebooks/figures/figure_controlled_releases.ipynb \
	notebooks/figures/cloudsen12_experiment.ipynb \
	notebooks/figures/ablation_threshold_pixels.ipynb


help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment
check-condaenv:  ## Verify the conda env exists (helpful error if not)
	@[ -x $(ENV_BIN)/python ] || { \
		printf "\033[1;31m>>> Conda env '$(CONDA_ENV)' not found. Run 'make condaenv' first.\033[0m\n"; \
		exit 1; }

condaenv:  ## 🐍 Create the marss2lpy312 conda env (idempotent) and install deps from the lock
	@[ -x $(ENV_BIN)/python ] || $(CONDA) create -y -n $(CONDA_ENV) python=$(VERSION) pip
	$(ENV_BIN)/pip install -q pip-tools
	@if [ -f $(LOCKFILE) ]; then \
		$(ENV_BIN)/pip install -r $(LOCKFILE) && $(ENV_BIN)/pip install -e . --no-deps; \
		printf "\033[1;32m>>> ✅ Env $(CONDA_ENV) ready\033[0m\n"; \
	else \
		printf "\033[1;33m>>> No lock file yet — run 'make lock', then re-run 'make condaenv'.\033[0m\n"; \
	fi

##@ Lock File
lock: check-condaenv  ## 🔒 Regenerate environment/requirements-test.lock (pip-tools, georeader pinned to 2.3.1)
	$(ENV_BIN)/pip install -q pip-tools
	# --only-binary=basemap: basemap's sdist build-deps (an ancient numpy) don't
	# build on Python 3.12, so resolve it from its wheel instead of the sdist.
	$(ENV_BIN)/pip-compile --strip-extras --extra test \
		-P georeader-spaceml==2.3.2 \
		--pip-args "--only-binary=basemap" \
		--output-file $(LOCKFILE) \
		pyproject.toml
	@printf "\033[1;33m>>> Lock regenerated — don't forget to commit $(LOCKFILE)\033[0m\n"

##@ Linting
# ruff-lint:  ## Lint Check using ruff
# 		ruff format ${PKGROOT}/

ruff-check-all: check-condaenv  ## Lint Check using ruff
		$(ENV_BIN)/ruff check --fix ${PKGROOT}/  --unsafe-fixes
		@printf "\033[1;34mruff-linting (missing imports) passes!\033[0m\n\n"

ruff-check-missing-imports: check-condaenv  ## Ruff Check for undefined functions
		$(ENV_BIN)/ruff check --fix ${PKGROOT}/ --select F821
		$(ENV_BIN)/ruff check --fix ${PKGROOT}/ --select E113

lint: ## Code styling - black, isort
		@printf "\033[1;34mRunning linting with ruff...\033[0m\n\n"
		make ruff-check-missing-imports
		@printf "\033[1;34mruff-linting (missing imports) passes!\033[0m\n\n"

##@ Formatting
black: check-condaenv  ## Format code in-place using black.
		@printf "\033[1;34mRunning formatting with Black...\033[0m\n\n"
		$(ENV_BIN)/black ${PKGROOT}/ -l 100 .
		@printf "\033[1;34mBlack passes...!\033[0m\n\n"

isort: check-condaenv  ## Format imports in-place using isort.
		@printf "\033[1;34mRunning formatting with isort...\033[0m\n\n"
		$(ENV_BIN)/isort ${PKGROOT}/
		@printf "\033[1;34misort passes...!\033[0m\n\n"

format: ## Code styling - black, isort
		@printf "\033[1;34mRunning formatting with Black and isort...\033[0m\n\n"
		make black
		make isort
		@printf "\033[1;34mPassed Formatting!\033[0m\n\n"


##@ Testing
test: check-condaenv  ## Test code using pytest.
	@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
	$(ENV_BIN)/pytest -v tests
	@printf "\033[1;34mPyTest passes!\033[0m\n\n"

test-cov: check-condaenv  ## Run tests with coverage report
	@printf "\033[1;34mRunning tests with coverage...\033[0m\n\n"
	$(ENV_BIN)/pytest --cov=marss2l --cov-report=term-missing -v tests

test-fast: check-condaenv  ## Run tests, stop on first failure, no warnings
	@printf "\033[1;34mRunning fast tests (failfast, no warnings)...\033[0m\n\n"
	$(ENV_BIN)/pytest -v -x -p no:warnings tests

test-file: check-condaenv  ## Run a specific test file: make test-file FILE=tests/test_plume_detection.py
	@printf "\033[1;34mRunning tests in file: $(FILE)\033[0m\n\n"
	$(ENV_BIN)/pytest -v $(FILE)

test-notebooks: check-condaenv  ## Run notebooks as integration tests with nbmake
	@printf "\033[1;34mRunning notebooks with nbmake...\033[0m\n\n"
	$(ENV_BIN)/pytest --nbmake -v --nbmake-timeout=600 --nbmake-kernel=$(NOTEBOOK_KERNEL) \
		$(NOTEBOOKS)
	@printf "\033[1;34mNotebook tests pass!\033[0m\n\n"

test-integration: check-condaenv  ## Run integration tests (train_final + notebooks), loading .env if present
	@printf "\033[1;34mRunning integration tests (loads .env if present)...\033[0m\n\n"
	@if [ -f .env ]; then \
		while IFS='=' read -r key val; do \
			case "$$key" in ''|\#*) continue;; esac; \
			key="$${key#export }"; \
			case "$$val" in \
				\"*\") val="$${val#\"}"; val="$${val%\"}";; \
				\'*\') val="$${val#\'}"; val="$${val%\'}";; \
			esac; \
			export "$$key=$$val"; \
		done < .env; \
	fi; \
	$(ENV_BIN)/pytest -v -m integration tests && \
	$(ENV_BIN)/pytest --nbmake -v --nbmake-timeout=600 --nbmake-kernel=$(NOTEBOOK_KERNEL) \
		$(NOTEBOOKS)
	@printf "\033[1;34mIntegration tests pass!\033[0m\n\n"

##@ Building
build: check-condaenv ## Build the marss2l package
	@printf "\033[1;34mBuilding package...\033[0m\n\n"
	rm -rf build/
	rm -rf dist/
	$(ENV_BIN)/python -m build

publish: check-condaenv ## Publish a release to PyPI
	@echo "🚀 Publishing: Dry run."
	$(ENV_BIN)/python -m twine check dist/*
	$(ENV_BIN)/python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose
	@echo "🚀 Publishing to PyPI."
	$(ENV_BIN)/python -m twine upload dist/*

##@ Documentation
.PHONY: docs-test
docs-test: check-condaenv ## Test if documentation can be built without warnings or errors
	$(ENV_BIN)/mkdocs build -s

.PHONY: docs
docs: check-condaenv ## Build and serve the documentation
	$(ENV_BIN)/mkdocs serve

.PHONY: docs-build
docs-build: check-condaenv ## Build the documentation
	$(ENV_BIN)/mkdocs build

.PHONY: docs-publish
docs-publish: check-condaenv ## Build and publish the documentation to GitHub Pages
	$(ENV_BIN)/mkdocs build
	$(ENV_BIN)/ghp-import -n -p -f site
