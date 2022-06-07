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
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.image.psnr import _psnr_compute, _psnr_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn


class PeakSignalNoiseRatio(Metric):
    r"""
    Computes `Computes Peak Signal-to-Noise Ratio`_ (PSNR):

    .. math:: \text{PSNR}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)}\right)

    Where :math:`\text{MSE}` denotes the `mean-squared-error`_ function.

    Args:
        data_range:
            the range of the data. If None, it is determined from the data (max - min).
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not given.

    Example:
        >>> from torchmetrics import PeakSignalNoiseRatio
        >>> psnr = PeakSignalNoiseRatio()
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)

    .. note::
        Half precision is only support on GPU for this metric

    """
    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    min_target: Tensor
    max_target: Tensor

    def __init__(
        self,
        data_range: Optional[float] = None,
        base: float = 10.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

        if dim is None:
            self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
            self.add_state("total", default=[], dist_reduce_fx="cat")

        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError("The `data_range` must be given when `dim` is not None.")

            self.data_range = None
            self.add_state("min_target", default=tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=tensor(0.0), dist_reduce_fx=torch.max)
        else:
            self.add_state("data_range", default=tensor(float(data_range)), dist_reduce_fx="mean")
        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, n_obs = _psnr_update(preds, target, dim=self.dim)
        if self.dim is None:
            if self.data_range is None:
                # keep track of min and max target values
                self.min_target = min(target.min(), self.min_target)
                self.max_target = max(target.max(), self.max_target)

            self.sum_squared_error += sum_squared_error
            self.total += n_obs
        else:
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(n_obs)

    def compute(self) -> Tensor:
        """Compute peak signal-to-noise ratio over state."""
        if self.data_range is not None:
            data_range = self.data_range
        else:
            data_range = self.max_target - self.min_target

        if self.dim is None:
            sum_squared_error = self.sum_squared_error
            total = self.total
        else:
            sum_squared_error = torch.cat([values.flatten() for values in self.sum_squared_error])
            total = torch.cat([values.flatten() for values in self.total])
        return _psnr_compute(sum_squared_error, total, data_range, base=self.base, reduction=self.reduction)
