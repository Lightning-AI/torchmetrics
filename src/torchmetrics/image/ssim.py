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
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ssim import _multiscale_ssim_update, _ssim_check_inputs, _ssim_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["StructuralSimilarityIndexMeasure.plot", "MultiScaleStructuralSimilarityIndexMeasure.plot"]


class StructuralSimilarityIndexMeasure(Metric):
    """Compute Structual Similarity Index Measure (SSIM_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of `forward` and `compute` the metric returns the following output

    - ``ssim`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average SSIM value
      over sample else returns tensor of shape ``(N,)`` with SSIM values per sample

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If ``True`` (default), a gaussian kernel is used, if ``False`` a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over individual batch scores

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.image import StructuralSimilarityIndexMeasure
        >>> preds = torch.rand([3, 3, 256, 256])
        >>> target = preds * 0.75
        >>> ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        >>> ssim(preds, target)
        tensor(0.9219)
    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx="cat")

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if return_contrast_sensitivity or return_full_image:
            self.add_state("image_return", default=[], dist_reduce_fx="cat")

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity_pack = _ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2,
            self.return_full_image,
            self.return_contrast_sensitivity,
        )

        if isinstance(similarity_pack, tuple):
            similarity, image = similarity_pack
        else:
            similarity = similarity_pack

        if self.return_contrast_sensitivity or self.return_full_image:
            self.image_return.append(image)

        if self.reduction in ("elementwise_mean", "sum"):
            self.similarity += similarity.sum()
            self.total += preds.shape[0]
        else:
            self.similarity.append(similarity)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute SSIM over state."""
        if self.reduction == "elementwise_mean":
            similarity = self.similarity / self.total
        elif self.reduction == "sum":
            similarity = self.similarity
        else:
            similarity = dim_zero_cat(self.similarity)

        if self.return_contrast_sensitivity or self.return_full_image:
            image_return = dim_zero_cat(self.image_return)
            return similarity, image_return

        return similarity

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
            >>> from torchmetrics.image import StructuralSimilarityIndexMeasure
            >>> preds = torch.rand([3, 3, 256, 256])
            >>> target = preds * 0.75
            >>> metric = StructuralSimilarityIndexMeasure(data_range=1.0)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image import StructuralSimilarityIndexMeasure
            >>> preds = torch.rand([3, 3, 256, 256])
            >>> target = preds * 0.75
            >>> metric = StructuralSimilarityIndexMeasure(data_range=1.0)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)


class MultiScaleStructuralSimilarityIndexMeasure(Metric):
    """Compute `MultiScaleSSIM`_, Multi-scale Structural Similarity Index Measure.

    This metric is is a generalization of Structural Similarity Index Measure by incorporating image details at
    different resolution scores.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of `forward` and `compute` the metric returns the following output

    - ``msssim`` (: :class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average MSSSIM
      value over sample else returns tensor of shape ``(N,)`` with SSIM values per sample

    Args:
        gaussian_kernel: If ``True`` (default), a gaussian kernel is used, if false a uniform kernel is used
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
            The ``data_range`` must be given when ``dim`` is not None.
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
        >>> from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
        >>> import torch
        >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        >>> ms_ssim(preds, target)
        tensor(0.9627)
    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        kernel_size: Union[int, Sequence[int]] = 11,
        sigma: Union[float, Sequence[float]] = 1.5,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        betas: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        normalize: Literal["relu", "simple", None] = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx="cat")

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

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

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity = _multiscale_ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2,
            self.betas,
            self.normalize,
        )

        if self.reduction in ("none", None):
            self.similarity.append(similarity)
        else:
            self.similarity += similarity.sum()

        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute MS-SSIM over state."""
        if self.reduction in ("none", None):
            return dim_zero_cat(self.similarity)
        if self.reduction == "sum":
            return self.similarity
        return self.similarity / self.total

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
            >>> from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
            >>> import torch
            >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
            >>> target = preds * 0.75
            >>> metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
            >>> import torch
            >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
            >>> target = preds * 0.75
            >>> metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
