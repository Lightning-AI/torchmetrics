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
import os
from typing import Any, ClassVar, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import Literal

from torchmetrics.functional.image.lpips import _LPIPS, _lpips_compute, _lpips_update, _NoTrainLpips
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE, _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["LearnedPerceptualImagePatchSimilarity.plot"]

if _TORCHVISION_AVAILABLE:

    def _download_lpips() -> None:
        _LPIPS(pretrained=True, net="vgg")

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_lpips):
        __doctest_skip__ = ["LearnedPerceptualImagePatchSimilarity", "LearnedPerceptualImagePatchSimilarity.plot"]
else:
    __doctest_skip__ = ["LearnedPerceptualImagePatchSimilarity", "LearnedPerceptualImagePatchSimilarity.plot"]


class LearnedPerceptualImagePatchSimilarity(Metric):
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates the perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    .. note:: using this metrics requires you to have ``lpips`` package installed. Either install
        as ``pip install torchmetrics[image]`` or ``pip install lpips``

    .. note:: this metric is not scriptable when using ``torch<1.8``. Please update your pytorch installation
        if this is a issue.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``img1`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, 3, H, W)``
    - ``img2`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, 3, H, W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``lpips`` (:class:`~torch.Tensor`): returns float scalar tensor with average LPIPS value over samples

    Args:
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``lpips`` package is not installed
        ValueError:
            If ``net_type`` is not one of ``"vgg"``, ``"alex"`` or ``"squeeze"``
        ValueError:
            If ``reduction`` is not one of ``"mean"`` or ``"sum"``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        >>> lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        >>> # LPIPS needs the images to be in the [-1, 1] range.
        >>> img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> lpips(img1, img2)
        tensor(0.1046, grad_fn=<SqueezeBackward0>)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    sum_scores: Tensor
    total: Tensor

    # due to the use of named tuple in the backbone the net variable cannot be scripted
    __jit_ignored_attributes__: ClassVar[List[str]] = ["net"]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _LPIPS_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that lpips is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install lpips`."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}.")
        self.net = _NoTrainLpips(net=net_type)

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` should be an bool but got {normalize}")
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update internal states with lpips score."""
        loss, total = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        self.sum_scores += loss.sum()
        self.total += total

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        return _lpips_compute(self.sum_scores, self.total, self.reduction)

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
            >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> metric.update(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
