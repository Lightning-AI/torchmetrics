# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.explained_variance import (
    _explained_variance_compute,
    _explained_variance_update,
)
from torchmetrics.metric import Metric


class ExplainedVariance(Metric):
    r"""
    Computes `explained variance`_:

    .. math:: \text{ExplainedVariance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Forward accepts

    - ``preds`` (float tensor): ``(N,)`` or ``(N, ...)`` (multioutput)
    - ``target`` (long tensor): ``(N,)`` or ``(N, ...)`` (multioutput)

    In the case of multioutput, as default the variances will be uniformly
    averaged over the additional dimensions. Please see argument `multioutput`
    for changing this behavior.

    Args:
        multioutput:
            Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is `'uniform_average'`.):

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Raises:
        ValueError:
            If ``multioutput`` is not one of ``"raw_values"``, ``"uniform_average"`` or ``"variance_weighted"``.

    Example:
        >>> from torchmetrics import ExplainedVariance
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance = ExplainedVariance()
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance = ExplainedVariance(multioutput='raw_values')
        >>> explained_variance(preds, target)
        tensor([0.9677, 1.0000])

    """
    n_obs: Tensor
    sum_error: Tensor
    sum_squared_error: Tensor
    sum_target: Tensor
    sum_squared_target: Tensor

    def __init__(
        self,
        multioutput: str = "uniform_average",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        allowed_multioutput = ("raw_values", "uniform_average", "variance_weighted")
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}"
            )
        self.multioutput: str = multioutput
        self.add_state("sum_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_target", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_target", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        n_obs, sum_error, sum_squared_error, sum_target, sum_squared_target = _explained_variance_update(preds, target)
        self.n_obs = self.n_obs + n_obs
        self.sum_error = self.sum_error + sum_error
        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.sum_target = self.sum_target + sum_target
        self.sum_squared_target = self.sum_squared_target + sum_squared_target

    def compute(self) -> Union[Tensor, Sequence[Tensor]]:
        """Computes explained variance over state."""
        return _explained_variance_compute(
            self.n_obs,
            self.sum_error,
            self.sum_squared_error,
            self.sum_target,
            self.sum_squared_target,
            self.multioutput,
        )

    @property
    def is_differentiable(self) -> bool:
        return True
