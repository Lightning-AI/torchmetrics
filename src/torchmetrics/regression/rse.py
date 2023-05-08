from typing import Any

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.r2 import _r2_score_update
from torchmetrics.functional.regression.rse import _relative_squared_error_compute
from torchmetrics.metric import Metric


class RelativeSquaredError(Metric):
    r"""Computes the (mean) relative squared error (RSE):

    .. math:: \text{RSE} = \frac{\sum_i^N(y_i - \hat{y_i})^2}{\sum_i^N(y_i - \overline{y})^2}
    Where :math:`y` is a tensor of target values with mean :math:`\overline{y}`, and
    :math:`\hat{y}` is a tensor of predictions.

    If num_outputs > 1, the returned value is averaged over all the outputs.

    Args:
        num_outputs: Number of outputs in multioutput setting
        squared: If True returns RSE value, if False returns RRSE value.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.regression import RelativeSquaredError
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> relative_squared_error = RelativeSquaredError()
        >>> relative_squared_error(preds, target)
        tensor(0.0514)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    sum_error: Tensor
    residual: Tensor
    total: Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("residual", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, sum_error, residual, total = _r2_score_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total

    def compute(self) -> Tensor:
        """Computes relative squared error over state."""
        return _relative_squared_error_compute(
            self.sum_squared_error, self.sum_error, self.residual, self.total, squared=self.squared
        )
