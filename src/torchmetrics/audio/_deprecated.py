from typing import Any, Callable, Optional

from typing_extensions import Literal

from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.utilities import _future_warning


class _PermutationInvariantTraining(PermutationInvariantTraining):
    def __init__(
        self,
        metric_func: Callable,
        eval_func: Literal["max", "min"] = "max",
        **kwargs: Any,
    ) -> None:
        _future_warning(
            "Importing `PermutationInvariantTraining` from TM root was deprecated and will be removed in 2.0; "
            "use importing from domain specific sub-package `torchmetrics.audio` instead."
        )
        return super().__init__(metric_func=metric_func, eval_func=eval_func, **kwargs)


class _ScaleInvariantSignalDistortionRatio(ScaleInvariantSignalDistortionRatio):
    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        _future_warning(
            "Importing `ScaleInvariantSignalDistortionRatio` from TM root was deprecated and will be removed in 2.0; "
            "use importing from domain specific sub-package `torchmetrics.audio` instead."
        )
        return super().__init__(zero_mean=zero_mean, **kwargs)


class _ScaleInvariantSignalNoiseRatio(ScaleInvariantSignalNoiseRatio):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        _future_warning(
            "Importing `ScaleInvariantSignalNoiseRatio` from TM root was deprecated and will be removed in 2.0; "
            "use importing from domain specific sub-package `torchmetrics.audio` instead."
        )
        return super().__init__(**kwargs)


class _SignalDistortionRatio(SignalDistortionRatio):
    def __init__(
        self,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        _future_warning(
            "Importing `SignalDistortionRatio` from TM root was deprecated and will be removed in 2.0; "
            "use importing from domain specific sub-package `torchmetrics.audio` instead."
        )
        return super().__init__(
            use_cg_iter=use_cg_iter, filter_length=filter_length, zero_mean=zero_mean, load_diag=load_diag, **kwargs
        )


class _SignalNoiseRatio(SignalNoiseRatio):
    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        _future_warning(
            "Importing `SignalNoiseRatio` from TM root was deprecated and will be removed in 2.0; "
            "use importing from domain specific sub-package `torchmetrics.audio` instead."
        )
        return super().__init__(zero_mean=zero_mean, **kwargs)
