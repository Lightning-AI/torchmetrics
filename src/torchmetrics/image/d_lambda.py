# Copyright The Lightning team.
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
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.d_lambda import (
    _spectral_distortion_index_band_uqi,
    _spectral_distortion_index_from_band_uqi,
    _spectral_distortion_index_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SpectralDistortionIndex.plot"]


class SpectralDistortionIndex(Metric):
    """Compute Spectral Distortion Index (SpectralDistortionIndex_) also now as D_lambda.

    The metric is used to compare the spectral distortion between two images.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Low resolution multispectral image of shape ``(N,C,H,W)``
    - ``target``(:class:`~torch.Tensor`): High resolution fused image of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``sdi`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average SDI value
      over sample else returns tensor of shape ``(N,)`` with SDI values per sample

    Args:
        p: Large spectral differences
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torch import rand
        >>> from torchmetrics.image import SpectralDistortionIndex
        >>> preds = rand([16, 3, 16, 16])
        >>> target = rand([16, 3, 16, 16])
        >>> sdi = SpectralDistortionIndex()
        >>> sdi(preds, target)
        tensor(0.0234)

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    target_band_uqi: Tensor
    preds_band_uqi: Tensor
    total: Tensor

    def __init__(
        self, p: int = 1, reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
        self.p = p
        allowed_reductions = ("elementwise_mean", "sum", "none")
        if reduction not in allowed_reductions:
            raise ValueError(f"Expected argument `reduction` be one of {allowed_reductions} but got {reduction}")
        self.reduction = reduction
        # Running sums over images of the per-image band-to-band UQI matrices. They start as scalars and take the
        # ``(C, C)`` shape of the first batch through broadcasting, since the number of bands is not known up front.
        self.add_state("target_band_uqi", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("preds_band_uqi", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with preds and target."""
        preds, target = _spectral_distortion_index_update(preds, target)
        if self.target_band_uqi.ndim == 2 and self.target_band_uqi.shape[-1] != preds.shape[1]:
            raise ValueError(
                "Expected the number of channels to be the same across updates."
                f" Got {preds.shape[1]} but previously {self.target_band_uqi.shape[-1]}."
            )
        self.target_band_uqi = self.target_band_uqi + _spectral_distortion_index_band_uqi(target).sum(dim=0)
        self.preds_band_uqi = self.preds_band_uqi + _spectral_distortion_index_band_uqi(preds).sum(dim=0)
        self.total = self.total + preds.shape[0]

    def compute(self) -> Tensor:
        """Compute and returns spectral distortion index."""
        m1 = self.target_band_uqi / self.total
        m2 = self.preds_band_uqi / self.total
        return _spectral_distortion_index_from_band_uqi(m1, m2, self.p, self.reduction)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torch import rand
            >>> from torchmetrics.image import SpectralDistortionIndex
            >>> preds = rand([16, 3, 16, 16])
            >>> target = rand([16, 3, 16, 16])
            >>> metric = SpectralDistortionIndex()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand
            >>> from torchmetrics.image import SpectralDistortionIndex
            >>> preds = rand([16, 3, 16, 16])
            >>> target = rand([16, 3, 16, 16])
            >>> metric = SpectralDistortionIndex()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
