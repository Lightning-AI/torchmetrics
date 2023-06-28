from typing import Any, Callable, Optional

from typing_extensions import Literal

from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.utilities.prints import _deprecated_root_import_class


class _PermutationInvariantTraining(PermutationInvariantTraining):
    """Wrapper for deprecated import.

    >>> import torch
    >>> from torchmetrics.functional import scale_invariant_signal_noise_ratio
    >>> _ = torch.manual_seed(42)
    >>> preds = torch.randn(3, 2, 5) # [batch, spk, time]
    >>> target = torch.randn(3, 2, 5) # [batch, spk, time]
    >>> pit = _PermutationInvariantTraining(scale_invariant_signal_noise_ratio,
    ...     mode="speaker-wise", eval_func="max")
    >>> pit(preds, target)
    tensor(-2.1065)
    """

    def __init__(
        self,
        metric_func: Callable,
        mode: Literal["speaker-wise", "permutation-wise"] = "speaker-wise",
        eval_func: Literal["max", "min"] = "max",
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("PermutationInvariantTraining", "audio")
        super().__init__(metric_func=metric_func, mode=mode, eval_func=eval_func, **kwargs)


class _ScaleInvariantSignalDistortionRatio(ScaleInvariantSignalDistortionRatio):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> si_sdr = _ScaleInvariantSignalDistortionRatio()
    >>> si_sdr(preds, target)
    tensor(18.4030)
    """

    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("ScaleInvariantSignalDistortionRatio", "audio")
        super().__init__(zero_mean=zero_mean, **kwargs)


class _ScaleInvariantSignalNoiseRatio(ScaleInvariantSignalNoiseRatio):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> si_snr = _ScaleInvariantSignalNoiseRatio()
    >>> si_snr(preds, target)
    tensor(15.0918)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("ScaleInvariantSignalNoiseRatio", "audio")
        super().__init__(**kwargs)


class _SignalDistortionRatio(SignalDistortionRatio):
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(1)
    >>> preds = torch.randn(8000)
    >>> target = torch.randn(8000)
    >>> sdr = _SignalDistortionRatio()
    >>> sdr(preds, target)
    tensor(-12.0589)
    >>> # use with pit
    >>> from torchmetrics.functional import signal_distortion_ratio
    >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
    >>> target = torch.randn(4, 2, 8000)
    >>> pit = _PermutationInvariantTraining(signal_distortion_ratio,
    ...     mode="speaker-wise", eval_func="max")
    >>> pit(preds, target)
    tensor(-11.6051)
    """

    def __init__(
        self,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("SignalDistortionRatio", "audio")
        super().__init__(
            use_cg_iter=use_cg_iter, filter_length=filter_length, zero_mean=zero_mean, load_diag=load_diag, **kwargs
        )


class _SignalNoiseRatio(SignalNoiseRatio):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> snr = _SignalNoiseRatio()
    >>> snr(preds, target)
    tensor(16.1805)
    """

    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("SignalNoiseRatio", "audio")
        super().__init__(zero_mean=zero_mean, **kwargs)
