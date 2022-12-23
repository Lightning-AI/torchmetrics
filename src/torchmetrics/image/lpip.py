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
from typing import Any, List

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import Literal

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

if _LPIPS_AVAILABLE:
    from lpips import LPIPS as _LPIPS
else:

    class _LPIPS(Module):  # type: ignore
        pass

    __doctest_skip__ = ["LearnedPerceptualImagePatchSimilarity", "LPIPS"]


class NoTrainLpips(_LPIPS):
    def train(self, mode: bool) -> "NoTrainLpips":
        """the network should not be able to be switched away from evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor, normalize: bool) -> bool:
    """check that input is a valid image to the network."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check


class LearnedPerceptualImagePatchSimilarity(Metric):
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) is used to judge the perceptual similarity between
    two images. LPIPS essentially computes the similarity between the activations of two image patches for some
    pre-defined network. This measure has been shown to match human perception well. A low LPIPS score means that
    image patches are perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``.
    The minimum size of `H, W` depends on the chosen backbone (see `net_type` arg).

    .. note:: using this metrics requires you to have ``lpips`` package installed. Either install
        as ``pip install torchmetrics[image]`` or ``pip install lpips``

    .. note:: this metric is not scriptable when using ``torch<1.8``. Please update your pytorch installation
        if this is a issue.

    As input to 'update' the metric accepts the following input:

    - ``img1``: tensor with images of shape ``(N, 3, H, W)``
    - ``img2``: tensor with images of shape ``(N, 3, H, W)``
    
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
        >>> lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        >>> # LPIPS needs the images to be in the [-1, 1] range.
        >>> img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> lpips(img1, img2)
        tensor(0.3493, grad_fn=<SqueezeBackward0>)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    # due to the use of named tuple in the backbone the net variable cannot be scripted
    __jit_ignored_attributes__ = ["net"]

    def __init__(
        self,
        net_type: str = "alex",
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
        self.net = NoTrainLpips(net=net_type, verbose=False)

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` should be an bool but got {normalize}")
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update internal states with lpips score."""
        if not (_valid_img(img1, self.normalize) and _valid_img(img2, self.normalize)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
                f" Got input with shape {img1.shape} and {img2.shape} and values in range"
                f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
                f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
            )
        loss = self.net(img1, img2, normalize=self.normalize).squeeze()
        self.sum_scores += loss.sum()
        self.total += img1.shape[0]

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        if self.reduction == "mean":
            return self.sum_scores / self.total
        if self.reduction == "sum":
            return self.sum_scores
