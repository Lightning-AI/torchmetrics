import torch
from torch import Tensor

from torchmetrics.functional.regression.wmape import (
    _weighted_mean_absolute_percentage_error_compute,
    _weighted_mean_absolute_percentage_error_update,
)
from torchmetrics.metric import Metric


class WeightedMeanAbsolutePercentageError(Metric):
    r"""
    Computes weighted absolute percentage error (`WAPE`_).

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()`` before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.

    Note:
        WAPE output is a non-negative floating point. Best result is 0.0 .

    Example:

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    sum_abs_error: Tensor
    sum_scale: Tensor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_scale", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.sum_scale += sum_scale

    def compute(self) -> Tensor:
        """Computes weighted absolute percentage error over state."""
        return _weighted_mean_absolute_percentage_error_compute(self.sum_abs_error, self.sum_scale)
