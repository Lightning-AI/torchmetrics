# AGENTS.md — Documentation

## Build Locally
- Install docs dependencies: pip install -r requirements/_docs.txt
- Optional full dev: pip install -e .[dev]
- Build:
  - From repo root: make -C docs html
  - Or: sphinx-build -b html docs/source docs/build/html

## Style & Content
- Follow Google/NumPy docstrings per repository config.
- Include shapes, dtypes, and small runnable examples in metric docs.
- Prefer relative cross-references; ensure links resolve.

## CI
- Keep docs warnings minimal; treat new warnings as errors locally before PR if possible.

## Tips
- Use examples/ to generate plots and embed in docs where appropriate.
- Large images: store under docs/source/_static/; reference relatively.
