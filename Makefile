.PHONY: conda black isort format ruff-check-all ruff-check-missing-imports lint test build publish mount-container help
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.12
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = marss2l


help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Linting
# ruff-lint:  ## Lint Check using ruff
# 		ruff format ${PKGROOT}/

ruff-check-all:  ## Lint Check using ruff
		ruff check --fix ${PKGROOT}/  --unsafe-fixes
		@printf "\033[1;34mruff-linting (missing imports) passes!\033[0m\n\n"

ruff-check-missing-imports:  ## Ruff Check for undefined functions
		ruff check --fix ${PKGROOT}/ --select F821
		ruff check --fix ${PKGROOT}/ --select E113

lint: ## Code styling - black, isort
		@printf "\033[1;34mRunning linting with ruff...\033[0m\n\n"
		make ruff-check-missing-imports
		@printf "\033[1;34mruff-linting (missing imports) passes!\033[0m\n\n"

##@ Formatting
black:  ## Format code in-place using black.
		@printf "\033[1;34mRunning formatting with Black...\033[0m\n\n"
		black ${PKGROOT}/ -l 100 .
		@printf "\033[1;34mBlack passes...!\033[0m\n\n"

isort:  ## Format imports in-place using isort.
		@printf "\033[1;34mRunning formatting with isort...\033[0m\n\n"
		isort ${PKGROOT}/ 
		@printf "\033[1;34misort passes...!\033[0m\n\n"

format: ## Code styling - black, isort
		@printf "\033[1;34mRunning formatting with Black and isort...\033[0m\n\n"
		make black
		make isort
		@printf "\033[1;34mPassed Formatting!\033[0m\n\n"

##@ Testing
test:  ## Test code using pytest.
		@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
		pytest -v tests
		@printf "\033[1;34mPyTest passes!\033[0m\n\n"

##@ Building
build: ## Build the marss2l package
	@printf "\033[1;34mBuilding package...\033[0m\n\n"
	rm -rf build/
	rm -rf dist/
	python -m build

publish: ## Publish a release to PyPI
	@echo "🚀 Publishing: Dry run."
	python -m twine check dist/*
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose
	@echo "🚀 Publishing to PyPI."
	python -m twine upload dist/*

##@ Documentation
.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	mkdocs serve

.PHONY: docs-build
docs-build: ## Build the documentation
	mkdocs build

.PHONY: docs-publish
docs-publish: ## Build and publish the documentation to GitHub Pages
	mkdocs build
	ghp-import -n -p -f site
