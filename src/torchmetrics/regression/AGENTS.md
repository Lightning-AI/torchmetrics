# AGENTS.md — Regression Metrics

## Scope
- Implements continuous regression metrics (MSE, MAE, R2, etc.) for tensors of shape [N, ...] (preds) and [N, ...] (target).

## Implementation Guidelines
- Input validation: same shape for preds and target; support float dtypes; document broadcasting if allowed.
- Numerical stability:
  - Prefer torch operations; avoid Python loops.
  - Use safe denominators with eps when dividing.
- Modules:
  - Accumulate sums and counts in update; compute final metric in compute.
  - Support DDP state synchronization via default Metric mechanisms.
- Functional:
  - No state; deterministic on given inputs.

## File Layout
- Module: src/torchmetrics/regression/<metric>.py
- Functional: src/torchmetrics/functional/regression/<metric>.py
- Tests: tests/regression/test_<metric>.py

## Testing
- Synthetic data covering typical/edge/error cases.
- Compare with NumPy/scikit-learn where possible.
- Device/dtype coverage; multi-batch accumulation equivalence.

## Lint & Style
- ruff check --fix ., black .
- Document units/scale expectations and permissible ranges.
