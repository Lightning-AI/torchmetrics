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
from typing import Any, List, Optional, Sequence, Tuple, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ssim import _multiscale_ssim_compute, _ssim_compute, _ssim_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class StructuralSimilarityIndexMeasure(Metric):
    """Computes Structual Similarity Index Measure (SSIM_).

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If ``True`` (default), a gaussian kernel is used, if ``False`` a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Tensor with SSIM score

    Example:
        >>> from torchmetrics import StructuralSimilarityIndexMeasure
        >>> import torch
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim = StructuralSimilarityIndexMeasure()
        >>> ssim(preds, target)
        tensor(0.9219)
    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SSIM` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

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
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.reduction,
            self.data_range,
            self.k1,
            self.k2,
            self.return_full_image,
            self.return_contrast_sensitivity,
        )


class MultiScaleStructuralSimilarityIndexMeasure(Metric):
    """Computes `MultiScaleSSIM`_, Multi-scale Structural Similarity Index Measure, which is a generalization of
    Structural Similarity Index Measure by incorporating image details at different resolution scores.

    Args:
        gaussian_kernel: If ``True`` (default), a gaussian kernel is used, if false a uniform kernel is used
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitivies returned by different image
            resolutions.
        normalize: When MultiScaleStructuralSimilarityIndexMeasure loss is used for training, it is desirable to use
            normalizes to improve the training stability. This `normalize` argument is out of scope of the original
            implementation [1], and it is adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Tensor with Multi-Scale SSIM score

    Raises:
        ValueError:
            If ``kernel_size`` is not an int or a Sequence of ints with size 2 or 3.
        ValueError:
            If ``betas`` is not a tuple of floats with lengt 2.
        ValueError:
            If ``normalize`` is neither `None`, `ReLU` nor `simple`.

    Example:
        >>> from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
        >>> import torch
        >>> preds = torch.rand([1, 1, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
        >>> ms_ssim(preds, target)
        tensor(0.9558)

    References:
        [1] Multi-Scale Structural Similarity For Image Quality Assessment by Zhou Wang, Eero P. Simoncelli and Alan C.
        Bovik `MultiScaleSSIM`_
    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        kernel_size: Union[int, Sequence[int]] = 11,
        sigma: Union[float, Sequence[float]] = 1.5,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        betas: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        normalize: Literal["relu", "simple", None] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `MS_SSIM` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        if not (isinstance(kernel_size, (Sequence, int))):
            raise ValueError(
                f"Argument `kernel_size` expected to be an sequence or an int, or a single int. Got {kernel_size}"
            )
        if isinstance(kernel_size, Sequence) and (
            len(kernel_size) not in (2, 3) or not all(isinstance(ks, int) for ks in kernel_size)
        ):
            raise ValueError(
                "Argument `kernel_size` expected to be an sequence of size 2 or 3 where each element is an int, "
                f"or a single int. Got {kernel_size}"
            )

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
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
            preds: Predictions from model of shape ``[N, C, H, W]``
            target: Ground truth values of shape ``[N, C, H, W]``
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
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.reduction,
            self.data_range,
            self.k1,
            self.k2,
            self.betas,
            self.normalize,
        )
