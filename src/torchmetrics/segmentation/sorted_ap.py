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

from torchmetrics.functional.segmentation.sorted_ap import (
    _sorted_ap_check_input,
    _sorted_ap_compute,
    _sorted_ap_update,
    _sorted_ap_validate_args,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SortedAP.plot"]


class SortedAP(Metric):
    r"""Compute the `SortedAP`_ metric for instance segmentation.

    The sortedAP metric is a threshold-free summary of average precision for instance segmentation. Instead
    of reporting AP at a fixed Jaccard (IoU) threshold, predictions and targets are matched at a very loose
    (fuzz) threshold, and the average precision is integrated over all possible thresholds by sweeping the
    IoU values of the matched pairs. The final score is the area under the resulting threshold-vs-precision
    curve, computed with the trapezoidal rule.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~list` of :class:`~torch.Tensor`): A list of predicted instance masks, one tensor
      of stacked binary masks of shape ``(num_preds, height, width)`` per image.
    - ``target`` (:class:`~list` of :class:`~torch.Tensor`): A list of target instance masks, one tensor
      of stacked binary masks of shape ``(num_targets, height, width)`` per image.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``sorted_ap`` (:class:`~torch.Tensor`): A scalar tensor with the sortedAP score in the range ``[0, 1]``.

    .. note::
        Unlike the semantic segmentation metrics in this module, ``SortedAP`` operates on **instance**
        masks. Instances need not be aligned or equal in number between ``preds`` and ``target`` -- the
        metric discovers the correspondence via matching. It does not accept logits/probabilities and has
        no ``num_classes``, ``include_background`` or ``input_format`` arguments.

    Args:
        match_method: The matching method used to assign predictions to targets. Can be one of
            ``"hungarian"`` (default, requires ``scipy``) or ``"greedy"``. If ``"hungarian"`` is
            requested but ``scipy`` is not installed, the metric falls back to ``"greedy"``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.segmentation import SortedAP
        >>> preds = [torch.tensor([[[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        ...                        [[0, 0, 0], [0, 1, 1], [0, 0, 0]]], dtype=torch.bool)]
        >>> target = [torch.tensor([[[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        ...                         [[0, 0, 0], [0, 1, 1], [0, 0, 0]]], dtype=torch.bool)]
        >>> metric = SortedAP()
        >>> metric(preds, target)
        tensor(1.)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    jaccard: list[Tensor]
    counts: Tensor

    def __init__(
        self,
        match_method: Literal["hungarian", "greedy"] = "hungarian",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _sorted_ap_validate_args(match_method)
        self.match_method = match_method

        self.add_state("jaccard", default=[], dist_reduce_fx="cat")
        self.add_state("counts", default=torch.zeros(3, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: list[Tensor], target: list[Tensor]) -> None:
        """Update the metric state with a batch of predictions and targets."""
        _sorted_ap_check_input(preds, target)
        # NOTE: _sorted_ap_update returns a 3-tuple (jaccard, counts, num_images); the third
        # value is currently unused by `compute`, so it is discarded here. (Fixed from the
        # original 2-value unpack, which raised `ValueError: too many values to unpack`.)
        jaccard, counts, _ = _sorted_ap_update(preds, target, self.match_method)
        if jaccard.numel() > 0:
            self.jaccard.append(jaccard.to(self.counts.device))
        self.counts = self.counts + counts.to(self.counts.device)

    def compute(self) -> Tensor:
        """Compute the sortedAP score over all accumulated state."""
        jaccard = dim_zero_cat(self.jaccard) if self.jaccard else torch.empty(0, device=self.counts.device)
        return _sorted_ap_compute(jaccard, self.counts)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of
                these results. If no value is provided, will automatically call ``metric.compute`` and plot
                that result.
            ax: An matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object

        """
        return self._plot(val, ax)
