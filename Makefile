PYTHON_FILE_PATHS = `(find . -iname "*.py" -not -path "./.venv/*")`

install: ## Install dependencies
	poetry install

hard-install: ## Clear and install dependencies
	rm -rf .venv && poetry install

requirements-export: ## Export build requirements
	poetry export -f requirements.txt --output requirements.txt --without-hashes

requirements-export-all: ## Export dev and build requirements
	poetry export --dev -f requirements.txt --output requirements.txt --without-hashes

upgrade-poetry: ## Upgrade poetry and dependencies
	poetry self update
	poetry run pip install --upgrade pip wheel setuptools
	poetry update

pylint: ## Run Pylint
	poetry run pylint -s no --rcfile=../.pylintrc $(PYTHON_FILE_PATHS)

isort: ## Run Isort
	poetry run isort --check-only $(PYTHON_FILE_PATHS)

isort-fix: ## Run Isort with automated fix
	poetry run isort $(PYTHON_FILE_PATHS)

black: ## Run Black
	poetry run black --check --quiet $(PYTHON_FILE_PATHS)

black-fix: ## Run Black with automated fix
	poetry run black $(PYTHON_FILE_PATHS)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

