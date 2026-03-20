# CLAUDE.md

回答は全て日本語で行い，思考は英語で行ってください
コメント文を書くときは説明の部分は日本語で行ってください

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python project named "rikka" using `uv` for package and dependency management. The project requires Python 3.14 and uses Pydantic for data validation.

## Development Commands

### Running the Application
```sh
uv run rikka
```

### Code Quality Commands

**Format code:**
```sh
uv run ruff format
```

**Lint code:**
```sh
uv run ruff check
```

**Lint with auto-fix:**
```sh
uv run ruff check --fix
```

**Type checking:**
```sh
uv run mypy src/
```

### Pre-commit

Install pre-commit hooks:
```sh
uv run pre-commit install
```

Run pre-commit on all files (useful when CI fails):
```sh
uv run pre-commit run --all-files
```

This command auto-fixes many issues and outputs errors that need manual fixing. Most CI failures can be resolved by pushing the auto-fixed changes.

### Building

Build the package:
```sh
uv build
```

### Dependency Management

Install all dependencies including dev dependencies:
```sh
uv sync --all-groups
```

## Project Structure

- `src/rikka/` - Main package directory
  - `__init__.py` - Package entry point, defines `main()` function
  - `main.py` - Contains utility functions

## Code Quality Configuration

### Ruff
- Line length: 88
- Target version: Python 3.13
- Enabled linters: pycodestyle (E/W), pyflakes (F), isort (I), pep8-naming (N), pyupgrade (UP), flake8-bugbear (B)
- Quote style: double quotes
- Indent style: spaces

### Mypy
- Python version: 3.12
- Strict mode enabled
- Uses Pydantic plugin for enhanced validation
- Configuration enforces typed Pydantic models with extra fields forbidden

## CI/CD

The project uses GitHub Actions for continuous integration:
- Runs on push to `main` and on pull requests
- Uses Python 3.14
- Executes pre-commit hooks on all files
- Builds the package to verify build succeeds

If CI fails, run `uv run pre-commit run --all-files` locally to reproduce and fix most issues.
