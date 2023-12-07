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

from typing import Any, Dict, List, Optional, Sequence, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.d_s import _spatial_distortion_index_compute, _spatial_distortion_index_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SpatialDistortionIndex.plot"]


class SpatialDistortionIndex(Metric):
    """Compute Spatial Distortion Index (SpatialDistortionIndex_) also now as D_s.

    The metric is used to compare the spatial distortion between two images.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): High resolution multispectral image of shape ``(N,C,H,W)``.
    - ``target`` (:class:`~Dict`): A dictionary containing the following keys:
        - ``ms`` (:class:`~torch.Tensor`): Low resolution multispectral image of shape ``(N,C,H',W')``.
        - ``pan`` (:class:`~torch.Tensor`): High resolution panchromatic image of shape ``(N,C,H,W)``.
        - ``pan_lr`` (:class:`~torch.Tensor`): Low resolution panchromatic image of shape ``(N,C,H',W')``.

    where H and W must be multiple of H' and W'.

    As output of `forward` and `compute` the metric returns the following output

    - ``sdi`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average SDI value
      over sample else returns tensor of shape ``(N,)`` with SDI values per sample

    Args:
        p: Order of the norm applied on the difference.
        ws: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.image import SpatialDistortionIndex
        >>> preds = torch.rand([16, 3, 32, 32])
        >>> target = {
        ...     'ms': torch.rand([16, 3, 16, 16]),
        ...     'pan': torch.rand([16, 3, 32, 32]),
        ... }
        >>> sdi = SpatialDistortionIndex()
        >>> sdi(preds, target)
        tensor(0.0090)

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    ms: List[Tensor]
    pan: List[Tensor]
    pan_lr: List[Tensor]

    def __init__(
        self,
        p: int = 1,
        ws: int = 7,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpatialDistortionIndex` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
        self.p = p
        if not isinstance(ws, int) or ws <= 0:
            raise ValueError(f"Expected `ws` to be a positive integer. Got ws: {ws}.")
        self.ws = ws
        allowed_reductions = ("elementwise_mean", "sum", "none")
        if reduction not in allowed_reductions:
            raise ValueError(f"Expected argument `reduction` be one of {allowed_reductions} but got {reduction}")
        self.reduction = reduction
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("ms", default=[], dist_reduce_fx="cat")
        self.add_state("pan", default=[], dist_reduce_fx="cat")
        self.add_state("pan_lr", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Dict[str, Tensor]) -> None:
        """Update state with preds and target."""
        preds, target = _spatial_distortion_index_update(preds, target)
        self.preds.append(preds)
        self.ms.append(target["ms"])
        self.pan.append(target["pan"])
        if "pan_lr" in target:
            self.pan_lr.append(target["pan_lr"])

    def compute(self) -> Tensor:
        """Compute and returns spatial distortion index."""
        preds = dim_zero_cat(self.preds)
        ms = dim_zero_cat(self.ms)
        pan = dim_zero_cat(self.pan)
        pan_lr = dim_zero_cat(self.pan_lr) if len(self.pan_lr) > 0 else None
        target = {
            "ms": ms,
            "pan": pan,
            **({"pan_lr": pan_lr} if pan_lr is not None else {}),
        }
        return _spatial_distortion_index_compute(preds, target, self.p, self.ws, self.reduction)

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
            >>> import torch
            >>> _ = torch.manual_seed(42)
            >>> from torchmetrics.image import SpatialDistortionIndex
            >>> preds = torch.rand([16, 3, 32, 32])
            >>> target = {
            ...     'ms': torch.rand([16, 3, 16, 16]),
            ...     'pan': torch.rand([16, 3, 32, 32]),
            ... }
            >>> metric = SpatialDistortionIndex()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> _ = torch.manual_seed(42)
            >>> from torchmetrics.image import SpatialDistortionIndex
            >>> preds = torch.rand([16, 3, 32, 32])
            >>> target = {
            ...     'ms': torch.rand([16, 3, 16, 16]),
            ...     'pan': torch.rand([16, 3, 32, 32]),
            ... }
            >>> metric = SpatialDistortionIndex()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)