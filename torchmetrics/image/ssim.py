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
from typing import Any, List, Optional, Sequence, Tuple

import torch
from deprecate import deprecated, void
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ssim import _multiscale_ssim_compute, _ssim_compute, _ssim_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class StructuralSimilarityIndexMeasure(Metric):
    """Computes Structual Similarity Index Measure (SSIM_).

    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.

    Return:
        Tensor with SSIM score

    Example:
        >>> from torchmetrics import StructuralSimilarityIndexMeasure
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim = StructuralSimilarityIndexMeasure()
        >>> ssim(preds, target)
        tensor(0.9219)
    """

    preds: List[Tensor]
    target: List[Tensor]
    higher_is_better = True

    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: str = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        rank_zero_warn(
            "Metric `SSIM` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _ssim_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _ssim_compute(
            preds, target, self.kernel_size, self.sigma, self.reduction, self.data_range, self.k1, self.k2
        )


class SSIM(StructuralSimilarityIndexMeasure):
    """Computes Structual Similarity Index Measure (SSIM_).

    .. deprecated:: v0.7
        Use :class:`torchmetrics.StructuralSimilarityIndexMeasure`. Will be removed in v0.8.

    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim = SSIM()
        >>> ssim(preds, target)
        tensor(0.9219)
    """

    @deprecated(target=StructuralSimilarityIndexMeasure, deprecated_in="0.7", remove_in="0.8")
    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: str = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        void(kernel_size, sigma, reduction, data_range, k1, k2, compute_on_step, dist_sync_on_step, process_group)


class MultiScaleStructuralSimilarityIndexMeasure(Metric):
    """Computes `MultiScaleSSIM`_, Multi-scale Structural Similarity Index Measure, which is a generalization of
    Structural Similarity Index Measure by incorporating image details at different resolution scores.

    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitivies returned by different image
            resolutions.
        normalize: When MultiScaleStructuralSimilarityIndexMeasure loss is used for training, it is desirable to use
            normalizes to improve the training stability. This `normalize` argument is out of scope of the original
            implementation [1], and it is adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.

    Return:
        Tensor with Multi-Scale SSIM score

    Example:
        >>> from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
        >>> preds = torch.rand([1, 1, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
        >>> ms_ssim(preds, target)
        tensor(0.9558)

    References:
    [1] Multi-Scale Structural Similarity For Image Quality Assessment by Zhou Wang, Eero P. Simoncelli and Alan C.
    Bovik `MultiScaleSSIM`_
    """

    preds: List[Tensor]
    target: List[Tensor]
    higher_is_better = True
    is_differentiable = True

    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: str = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        betas: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        normalize: Optional[Literal["relu", "simple"]] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        rank_zero_warn(
            "Metric `MS_SSIM` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        all_kernel_ints = all(isinstance(ks, int) for ks in kernel_size)
        if not isinstance(kernel_size, Sequence) or len(kernel_size) != 2 or not all_kernel_ints:
            raise ValueError(
                "Argument `kernel_size` expected to be an sequence of size 2 where each element is an int"
                f" but got {kernel_size}"
            )
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction
        if not isinstance(betas, tuple):
            raise ValueError("Argument `betas` is expected to be of a type tuple.")
        if isinstance(betas, tuple) and not all(isinstance(beta, float) for beta in betas):
            raise ValueError("Argument `betas` is expected to be a tuple of floats.")
        self.betas = betas
        if normalize and normalize not in ("relu", "simple"):
            raise ValueError("Argument `normalize` to be expected either `None` or one of 'relu' or 'simple'")
        self.normalize = normalize

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model of shape `[N, C, H, W]`
            target: Ground truth values of shape `[N, C, H, W]`
        """
        preds, target = _ssim_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _multiscale_ssim_compute(
            preds,
            target,
            self.kernel_size,
            self.sigma,
            self.reduction,
            self.data_range,
            self.k1,
            self.k2,
            self.betas,
            self.normalize,
        )
