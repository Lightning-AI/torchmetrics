# Copyright The Lightning team.

import pytest
import torch

from torchmetrics.functional.regression.symmetric_mape import symmetric_mean_absolute_percentage_error
from torchmetrics.regression.symmetric_mape import SymmetricMeanAbsolutePercentageError


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_symmetric_mape_large_finite_half_precision_values(dtype: torch.dtype) -> None:
    """Finite half-precision inputs must produce a finite score across updates."""
    preds = torch.tensor([40000.0, 0.0], dtype=dtype)
    target = torch.tensor([-40000.0, 0.0], dtype=dtype)

    actual = symmetric_mean_absolute_percentage_error(preds, target)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, torch.tensor(1.0))

    metric = SymmetricMeanAbsolutePercentageError()
    metric.update(preds[:1], target[:1])
    metric.update(preds[1:], target[1:])
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))
