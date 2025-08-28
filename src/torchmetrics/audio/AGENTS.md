# AGENTS.md — Audio Metrics

## Scope

- Audio signal quality and intelligibility metrics (e.g., SI-SDR, PESQ if available, STOI where licensing permits).

## Implementation Guidelines

- Inputs:
  - Typically [N, T] or [N, C, T]; document channel handling.
  - Sample rates required for some metrics; validate and document.
- Performance:
  - Prefer vectorized ops; avoid per-sample Python loops.
- Modules/Functional:
  - Same split as project convention; no state in functional.

## Dependencies

- Install extras: pip install torchmetrics[audio]
- Some metrics may require optional packages; fail with clear ModuleNotFoundError messages.

## Tests

- Synthetic test signals (sine, noise) to check basic invariants.
- Device/dtype coverage and shape validation.

## Lint & Style

- ruff check --fix ., black .
- Clear docstrings with shapes, sample rates, and units.
