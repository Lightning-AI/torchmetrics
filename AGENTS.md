# AGENTS.md — TorchMetrics Project Root

## 🛠️ Installation

- MUST install TorchMetrics via pip or conda unless project requirements explicitly override.
  - pip: `pip install torchmetrics`
  - conda: `conda install -c conda-forge torchmetrics`
- SHOULD use PyTorch >=2.0.0 for full compatibility (per requirements/base.txt).
- MAY install optional domains via:
  - Audio: `pip install torchmetrics[audio]`
  - Image: `pip install torchmetrics[image]`
  - Text: `pip install torchmetrics[text]`
  - Visual/Plotting: `pip install torchmetrics[visual]`

## ⚡ Development Environment

- MUST configure dev environment using `.devcontainer/devcontainer.json` for uniform development, especially with GPU.
- SHOULD use pre-commit hooks and VSCode if available for lint and autoformatting.
- Agents MUST NOT modify or delete container configs without explicit requests.

## ✅ Testing & Coverage

- MUST use `pytest` for all Python tests, launched from the repo root: `pytest tests`
- For full coverage: `pytest --cov=torchmetrics`
- MUST validate both functional (`torchmetrics/functional/`) and module (`torchmetrics/<domain>/`) interfaces.
- SHOULD pass tests locally before PR or CI runs.

## 🧹 Linting & Formatting

- MUST run all of:
  - `ruff format .` (autoformatting)
  - `ruff check . --fix` (linting)
  - `pre-commit run --all-files`
- Agents MAY reference `.pre-commit-config.yaml` for hook settings.

## 🔁 Continuous Integration (CI)

- Agents MUST NOT introduce non-reproducible steps in workflows.
- Support for DistributedDataParallel (DDP) and DataParallel (DP) must be preserved.
- Add/get workflows in `.github/workflows/`, following YAML syntax from official TorchMetrics CI.
- CI and docs build pipelines are triggered on pushes and PRs. See `.github/workflows/` for templates.

## 🤝 Contribution Guidelines

- New metrics must:
  - Split logic: functional in `torchmetrics/functional/<domain>/`, class/module in `torchmetrics/<domain>/`
  - Add tests in `tests/<domain>/test_<metric>.py` covering both interfaces, and DDP/multi-device.
  - Include docstrings, typing annotations, sample code in Google or NumPy style.
- PRs:
  - MUST pass lint and all tests before review.
  - Title best practice: `[domain] <description>`.
  - Issues and feature discussions: open in GitHub issues, reference relevant file paths.
- Agents SHOULD reference subdirectory AGENTS.md for domain/module rules.

## 📚 Reference

- For domain-specific instructions: see AGENTS.md in `torchmetrics/audio/`, `torchmetrics/image/`, `torchmetrics/text/` etc.
- General metric implementation: see [official documentation](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html).
