# AGENTS.md — Image Metrics

## Scope
- Image quality and similarity metrics (e.g., PSNR, SSIM, LPIPS when applicable).

## Implementation Guidelines
- Inputs:
  - Expect shapes [N, C, H, W] unless documented otherwise.
  - dtypes: float tensors scaled appropriately (0-1 or 0-255 as documented).
- Channels:
  - Support both grayscale and RGB; document requirements and conversions.
- Modules vs Functional:
  - Same separation as elsewhere; modules manage accumulation, functional is single-call.
- Device:
  - Avoid CPU/GPU transfers; move constants to device of inputs.

## Dependencies
- Install extras: pip install torchmetrics[image]
- If optional third-party backends are needed, raise informative errors when missing.

## Tests
- Include small synthetic tensors with known results.
- Consistency across devices and batches; test numeric tolerance.

## Lint & Style
- ruff check --fix ., black .
- Docstrings include expected ranges and example usage.
