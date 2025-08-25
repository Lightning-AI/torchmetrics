# AGENTS.md — Detection Metrics

## Scope

- Object detection metrics (e.g., mAP across IoU thresholds) and overlap utilities.

## Implementation Guidelines

- Inputs:
  - Predictions: boxes [N, num_preds, 4], scores [N, num_preds], labels [N, num_preds]
  - Targets: boxes [N, num_targets, 4], labels [N, num_targets]
  - Coordinate format: xyxy or cxcywh; document and validate.
- IoU/Area:
  - Vectorize IoU calculations; avoid nested Python loops.
- Thresholding:
  - Support configurable IoU thresholds and per-class averaging strategies.

## Dependencies

- Install extras: pip install torchmetrics[detection]

## Modules/Functional

- Functional implements per-batch computations.
- Modules aggregate across dataset for mAP; ensure correct DDP synchronization.

## Tests

- Synthetic small scenes with exact box overlaps to assert correctness.
- Class-agnostic and per-class evaluation; error handling for invalid shapes.

## Lint & Style

- ruff check --fix ., black .
- Docstrings with clear coordinate conventions and thresholds.
