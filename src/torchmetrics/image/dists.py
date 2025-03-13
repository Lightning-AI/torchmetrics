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
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from torchmetrics.functional.image.dists import _dists_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DeepImageStructureAndTextureSimilarity.plot"]

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["DeepImageStructureAndTextureSimilarity", "DeepImageStructureAndTextureSimilarity.plot"]


class DeepImageStructureAndTextureSimilarity(Metric):
    """Calculates Deep Image Structure and Texture Similarity (DISTS) score.

    The metric is a full-reference image quality assessment (IQA) model that combines sensitivity to structural
    distortions (e.g., artifacts due to noise, blur, or compression) with a tolerance of texture resampling
    (exchanging the content of a texture region with a new sample of the same texture). The metric is based on
    a convolutional neural network (CNN) that transforms the reference and distorted images to a new representation.
    Within this representation, a set of measurements are developed that are sufficient to capture the appearance
    of a variety of different visual distortions.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, 3, H, W)``
    - ``target`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, 3, H, W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``lpips`` (:class:`~torch.Tensor`): returns float scalar tensor with average LPIPS value over samples

    Args:
        reduction: specifies the reduction to apply to the output.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If `reduction` is not one of ["mean", "sum"]

    Example:
        >>> from torch import rand
        >>> from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
        >>> metric = DeepImageStructureAndTextureSimilarity()
        >>> preds = rand(10, 3, 100, 100)
        >>> target = rand(10, 3, 100, 100)
        >>> metric(preds, target)
        tensor(0.1882, grad_fn=<CloneBackward0>)

    """

    score: Tensor
    total: Tensor

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    def __init__(self, reduction: Optional[Literal["mean", "sum"]] = "mean", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        allowed_reductions = ("mean", "sum")
        if reduction not in allowed_reductions:
            raise ValueError(f"Argument `reduction` expected to be one of {allowed_reductions} but got {reduction}")
        self.reduction = reduction
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric state."""
        scores = _dists_update(preds, target)
        self.score += scores.sum()
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Computes the DISTS score."""
        return self.score / self.total if self.reduction == "mean" else self.score

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
            >>> from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
            >>> metric = DeepImageStructureAndTextureSimilarity()
            >>> metric.update(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
            >>> metric = DeepImageStructureAndTextureSimilarity()
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
