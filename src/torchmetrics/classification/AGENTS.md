# AGENTS.md — Classification Metrics

## Scope

- Implements classification metrics for binary, multiclass, and multilabel tasks.
- Provides both functional APIs (torchmetrics/functional/classification/) and Metric modules (torchmetrics/classification/).

## Implementation Guidelines

- Task API: expose a "task" parameter with values: "binary", "multiclass", "multilabel" where applicable.
- Inputs:
  - binary: preds shape [N] or [N, ...], target same; threshold default 0.5 for probabilities/logits.
  - multiclass: preds shape [N, C, ...] (logits/probs), target [N, ...] with class indices; support top_k where relevant.
  - multilabel: preds [N, C, ...], target [N, C, ...]; threshold default 0.5.
- Common args: num_classes (multiclass/multilabel), average ("micro"/"macro"/"weighted"/"none"), ignore_index, top_k.
- Modules:
  - Stateless math in update/compute; avoid heavy Python loops.
  - Support DDP reductions through Metric base class states; set sync_on_compute where needed.
  - full_state_update=False if batch-wise accumulation is safe; True if final pass over all data needed.
- Functional:
  - No internal state; single-call semantics.
  - Match module numerics and docstrings.

## File Layout

- Module: src/torchmetrics/classification/<metric>.py
- Functional: src/torchmetrics/functional/classification/<metric>.py
- Tests: tests/classification/test\_<metric>.py

## Testing

- Compare against known references (e.g., scikit-learn) when applicable.
- Parametrize over tasks, threshold, average, top_k, device, dtype, shape.
- DDP/single-device via pytest markers per repo conventions.

## Dependencies

- Base install covers most use; for developers, consider: pip install torchmetrics[all]

## Lint & Style

- ruff check --fix ., black .
- Numpy/Google docstring; clear examples; state shapes and dtypes explicitly.
