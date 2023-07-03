from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from typing_extensions import Literal

from torchmetrics.image.d_lambda import SpectralDistortionIndex
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.rase import RelativeAverageSpectralError
from torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
from torchmetrics.image.tv import TotalVariation
from torchmetrics.image.uqi import UniversalImageQualityIndex
from torchmetrics.utilities.prints import _deprecated_root_import_class


class _ErrorRelativeGlobalDimensionlessSynthesis(ErrorRelativeGlobalDimensionlessSynthesis):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))
    >>> target = preds * 0.75
    >>> ergas = _ErrorRelativeGlobalDimensionlessSynthesis()
    >>> torch.round(ergas(preds, target))
    tensor(154.)
    """

    def __init__(
        self,
        ratio: Union[int, float] = 4,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("ErrorRelativeGlobalDimensionlessSynthesis", "image")
        super().__init__(ratio=ratio, reduction=reduction, **kwargs)


class _MultiScaleStructuralSimilarityIndexMeasure(MultiScaleStructuralSimilarityIndexMeasure):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
    >>> target = preds * 0.75
    >>> ms_ssim = _MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    >>> ms_ssim(preds, target)
    tensor(0.9627)
    """

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
        _deprecated_root_import_class("MultiScaleStructuralSimilarityIndexMeasure", "image")
        super().__init__(
            gaussian_kernel=gaussian_kernel,
            kernel_size=kernel_size,
            sigma=sigma,
            reduction=reduction,
            data_range=data_range,
            k1=k1,
            k2=k2,
            betas=betas,
            normalize=normalize,
            **kwargs,
        )


class _PeakSignalNoiseRatio(PeakSignalNoiseRatio):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> psnr = _PeakSignalNoiseRatio()
    >>> preds = tensor([[0.0, 1.0], [2.0, 3.0]])
    >>> target = tensor([[3.0, 2.0], [1.0, 0.0]])
    >>> psnr(preds, target)
    tensor(2.5527)
    """

    def __init__(
        self,
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        base: float = 10.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("PeakSignalNoiseRatio", "image")
        super().__init__(data_range=data_range, base=base, reduction=reduction, dim=dim, **kwargs)


class _RelativeAverageSpectralError(RelativeAverageSpectralError):
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(22)
    >>> preds = torch.rand(4, 3, 16, 16)
    >>> target = torch.rand(4, 3, 16, 16)
    >>> rase = _RelativeAverageSpectralError()
    >>> rase(preds, target)
    tensor(5114.6641)
    """

    def __init__(
        self,
        window_size: int = 8,
        **kwargs: Dict[str, Any],
    ) -> None:
        _deprecated_root_import_class("RelativeAverageSpectralError", "image")
        super().__init__(window_size=window_size, **kwargs)


class _RootMeanSquaredErrorUsingSlidingWindow(RootMeanSquaredErrorUsingSlidingWindow):
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(22)
    >>> preds = torch.rand(4, 3, 16, 16)
    >>> target = torch.rand(4, 3, 16, 16)
    >>> rmse_sw = RootMeanSquaredErrorUsingSlidingWindow()
    >>> rmse_sw(preds, target)
    tensor(0.3999)
    """

    def __init__(
        self,
        window_size: int = 8,
        **kwargs: Dict[str, Any],
    ) -> None:
        _deprecated_root_import_class("RootMeanSquaredErrorUsingSlidingWindow", "image")
        super().__init__(window_size=window_size, **kwargs)


class _SpectralAngleMapper(SpectralAngleMapper):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(42))
    >>> target = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(123))
    >>> sam = _SpectralAngleMapper()
    >>> sam(preds, target)
    tensor(0.5943)
    """

    def __init__(
        self,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("SpectralAngleMapper", "image")
        super().__init__(reduction=reduction, **kwargs)


class _SpectralDistortionIndex(SpectralDistortionIndex):
    """Wrapper for deprecated import.

    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> preds = torch.rand([16, 3, 16, 16])
    >>> target = torch.rand([16, 3, 16, 16])
    >>> sdi = _SpectralDistortionIndex()
    >>> sdi(preds, target)
    tensor(0.0234)
    """

    def __init__(
        self, p: int = 1, reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean", **kwargs: Any
    ) -> None:
        _deprecated_root_import_class("SpectralDistortionIndex", "image")
        super().__init__(p=p, reduction=reduction, **kwargs)


class _StructuralSimilarityIndexMeasure(StructuralSimilarityIndexMeasure):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([3, 3, 256, 256])
    >>> target = preds * 0.75
    >>> ssim = _StructuralSimilarityIndexMeasure(data_range=1.0)
    >>> ssim(preds, target)
    tensor(0.9219)
    """

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
        _deprecated_root_import_class("StructuralSimilarityIndexMeasure", "image")
        super().__init__(
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction=reduction,
            data_range=data_range,
            k1=k1,
            k2=k2,
            return_full_image=return_full_image,
            return_contrast_sensitivity=return_contrast_sensitivity,
            **kwargs,
        )


class _TotalVariation(TotalVariation):
    """Wrapper for deprecated import.

    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> tv = _TotalVariation()
    >>> img = torch.rand(5, 3, 28, 28)
    >>> tv(img)
    tensor(7546.8018)
    """

    def __init__(self, reduction: Literal["mean", "sum", "none", None] = "sum", **kwargs: Any) -> None:
        _deprecated_root_import_class("TotalVariation", "image")
        super().__init__(reduction=reduction, **kwargs)


class _UniversalImageQualityIndex(UniversalImageQualityIndex):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([16, 1, 16, 16])
    >>> target = preds * 0.75
    >>> uqi = _UniversalImageQualityIndex()
    >>> uqi(preds, target)
    tensor(0.9216)
    """

    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("UniversalImageQualityIndex", "image")
        super().__init__(kernel_size=kernel_size, sigma=sigma, reduction=reduction, **kwargs)
