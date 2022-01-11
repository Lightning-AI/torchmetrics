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
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

if _LPIPS_AVAILABLE:
    from lpips import LPIPS as Lpips_backbone
else:

    class Lpips_backbone(torch.nn.Module):  # type: ignore
        pass


class NoTrainLpips(Lpips_backbone):
    def train(self, mode: bool) -> "NoTrainLpips":
        """the network should not be able to be switched away from evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor) -> bool:
    """check that input is a valid image to the network."""
    return img.ndim == 4 and img.shape[1] == 3 and img.min() >= -1.0 and img.max() <= 1.0


class LPIPS(Metric):
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) is used to judge the perceptual similarity between
    two images. LPIPS essentially computes the similarity between the activations of two image patches for some
    pre-defined network. This measure have been shown to match human perseption well. A low LPIPS score means that
    image patches are perceptual similar.

    Both input image patches are expected to have shape `[N, 3, H, W]` and be normalized to the [-1,1]
    range. The minimum size of `H, W` depends on the chosen backbone (see `net_type` arg).

    .. note:: using this metrics requires you to have ``lpips`` package installed. Either install
        as ``pip install torchmetrics[image]`` or ``pip install lpips``

    .. note:: this metric is not scriptable when using ``torch<1.8``. Please update your pytorch installation
        if this is a issue.

    Args:
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

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
        >>> from torchmetrics.image.lpip_similarity import LPIPS
        >>> lpips = LPIPS(net_type='vgg')
        >>> img1 = torch.rand(10, 3, 100, 100)
        >>> img2 = torch.rand(10, 3, 100, 100)
        >>> lpips(img1, img2)
        tensor(0.3566, grad_fn=<SqueezeBackward0>)
    """

    is_differentiable = True
    higher_is_better = False
    real_features: List[Tensor]
    fake_features: List[Tensor]

    # due to the use of named tuple in the backbone the net variable cannot be scriptet
    __jit_ignored_attributes__ = ["net"]

    def __init__(
        self,
        net_type: str = "alex",
        reduction: str = "mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

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

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update internal states with lpips score.

        Args:
            img1: tensor with images of shape [N, 3, H, W]
            img2: tensor with images of shape [N, 3, H, W]
        """
        if not (_valid_img(img1) and _valid_img(img2)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors (all values in range [-1,1])"
                f" and to have shape [N, 3, H, W] but `img1` have shape {img1.shape} with values in"
                f" range {[img1.min(), img1.max()]} and `img2` have shape {img2.shape} with value"
                f" in range {[img2.min(), img2.max()]}"
            )

        loss = self.net(img1, img2).squeeze()
        self.sum_scores += loss.sum()
        self.total += img1.shape[0]

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        if self.reduction == "mean":
            return self.sum_scores / self.total
        if self.reduction == "sum":
            return self.sum_scores
