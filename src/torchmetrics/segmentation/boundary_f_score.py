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

from torchmetrics.functional.segmentation.boundary_f_score import (
    _boundary_f_score_compute,
    _boundary_f_score_update,
    _boundary_f_score_validate_args,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BoundaryFScore.plot"]


class BoundaryFScore(Metric):
    r"""Compute the Boundary F-score for semantic segmentation.

    Boundary F-score evaluates how well predicted object contours align with target contours. A predicted boundary
    pixel counts as correct if a target boundary pixel exists within ``boundary_width`` pixels, and vice versa. The
    final score is the harmonic mean of boundary precision and boundary recall. The tolerance is expressed in pixels
    for 2D masks and voxels for 3D volumes. Classes that are absent in both prediction and target are ignored in
    reduced outputs and return ``nan`` when reported per class.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
          the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
          can be provided, where the integer values correspond to the class index. The input type can be controlled
          with the ``input_format`` argument.
        - ``target`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
          the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
          can be provided, where the integer values correspond to the class index. The input type can be controlled
          with the ``input_format`` argument.

    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``boundary_f_score`` (:class:`~torch.Tensor`): The boundary F-score. If ``per_class`` is set to ``True``,
          the output will be a tensor of shape ``(C,)`` with one score per class. If ``per_class`` is set to
          ``False``, the output will be a scalar tensor.

    Args:
        num_classes: The number of classes in the segmentation problem.
        include_background: Whether to include the background class in the computation.
        per_class: Whether to compute the score for each class separately, else average over all valid classes.
        boundary_width: Integer pixel tolerance used when matching predicted and target boundaries.
        input_format: What kind of input the function receives.
            Choose between ``"one-hot"`` for one-hot encoded tensors, ``"index"`` for index tensors
            or ``"mixed"`` for one one-hot encoded and one index tensor.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.segmentation import BoundaryFScore
        >>> preds = torch.zeros(2, 3, 8, 8, dtype=torch.int)
        >>> target = torch.zeros(2, 3, 8, 8, dtype=torch.int)
        >>> preds[:, 1, 2:6, 2:6] = 1
        >>> target[:, 1, 2:6, 2:6] = 1
        >>> metric = BoundaryFScore(num_classes=3)
        >>> metric(preds, target)
        tensor(1.)

    """

    score: Tensor
    samples: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        boundary_width: int = 1,
        input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _boundary_f_score_validate_args(num_classes, include_background, per_class, boundary_width, input_format)
        self.num_classes = num_classes
        self.include_background = include_background
        self.per_class = per_class
        self.boundary_width = boundary_width
        self.input_format = input_format

        num_classes = num_classes - 1 if not include_background else num_classes
        self.add_state("score", default=torch.zeros(num_classes if per_class else 1), dist_reduce_fx="sum")
        self.add_state(
            "samples",
            default=torch.zeros(num_classes if per_class else 1, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new data."""
        score, valid = _boundary_f_score_update(
            preds,
            target,
            num_classes=self.num_classes,
            include_background=self.include_background,
            boundary_width=self.boundary_width,
            input_format=self.input_format,
        )
        if self.per_class:
            self.score += torch.nan_to_num(score, nan=0.0).sum(dim=0)
            self.samples += valid.sum(dim=0)
            return

        reduced_score = _boundary_f_score_compute(score, valid, per_class=False)
        valid_samples = valid.any(dim=-1)
        self.score += torch.nan_to_num(reduced_score, nan=0.0).sum().unsqueeze(0)
        self.samples += valid_samples.sum().unsqueeze(0)

    def compute(self) -> Tensor:
        """Compute the final boundary F-score."""
        score = _safe_divide(self.score, self.samples, zero_division="nan")
        return score if self.per_class else score.squeeze()

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
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

            >>> import torch
            >>> from torchmetrics.segmentation import BoundaryFScore
            >>> metric = BoundaryFScore(num_classes=2)
            >>> preds = torch.zeros(2, 2, 16, 16, dtype=torch.int)
            >>> target = torch.zeros(2, 2, 16, 16, dtype=torch.int)
            >>> preds[:, 1, 4:12, 4:12] = 1
            >>> target[:, 1, 4:12, 4:12] = 1
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        """
        return self._plot(val, ax)
