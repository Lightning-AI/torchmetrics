# AGENTS.md — Devcontainer

## Usage
- Open the repo in VS Code and "Reopen in Container".
- GPU: ensure the host has NVIDIA Container Toolkit configured if using CUDA images.

## Customization
- Edit .devcontainer/devcontainer.json to change extensions, mounts, or container features.
- Edit .devcontainer/Dockerfile to change base image, CUDA version, or system packages.

## Best Practices
- Do not remove devcontainer config without explicit request.
- Keep Python version >=3.9 to match setup.py.
- Match PyTorch and CUDA versions with requirements/base.txt and local hardware.

## Tooling Inside Container
- Run pre-commit hooks: pre-commit run --all-files
- Lint/format: ruff check --fix ., black .
- Tests: pytest tests -q
