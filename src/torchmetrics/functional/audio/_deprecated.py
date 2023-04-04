from typing import Any, Callable, Optional, Tuple

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.utilities import _future_warning


def _permutation_invariant_training(
    preds: Tensor, target: Tensor, metric_func: Callable, eval_func: Literal["max", "min"] = "max", **kwargs: Any
) -> Tuple[Tensor, Tensor]:
    _future_warning(
        "Importing as `from torchmetrics.functional import permutation_invariant_training` was deprecated"
        " and will be removed in 2.0; use importing from domain specific sub-package `torchmetrics.audio` instead."
    )
    return permutation_invariant_training(
        preds=preds, target=target, metric_func=metric_func, eval_func=eval_func, **kwargs
    )


def _pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    _future_warning(
        "Importing as `from torchmetrics.functional import pit_permutate` was deprecated and will be removed in 2.0;"
        " use importing from domain specific sub-package `torchmetrics.audio` instead."
    )
    return pit_permutate(preds=preds, perm=perm)


def _scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    _future_warning(
        "Importing as `from torchmetrics.functional import pit_permutate` was deprecated and will be removed in 2.0;"
        " use importing from domain specific sub-package `torchmetrics.audio` instead."
    )
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=zero_mean)


def _signal_distortion_ratio(
    preds: Tensor,
    target: Tensor,
    use_cg_iter: Optional[int] = None,
    filter_length: int = 512,
    zero_mean: bool = False,
    load_diag: Optional[float] = None,
) -> Tensor:
    _future_warning(
        "Importing as `from torchmetrics.functional import signal_distortion_ratio` was deprecated"
        " and will be removed in 2.0; use importing from domain specific sub-package `torchmetrics.audio` instead."
    )
    return signal_distortion_ratio(
        preds=preds,
        target=target,
        use_cg_iter=use_cg_iter,
        filter_length=filter_length,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )


def _scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    _future_warning(
        "Importing as `from torchmetrics.functional import scale_invariant_signal_noise_ratio` was deprecated"
        " and will be removed in 2.0; use importing from domain specific sub-package `torchmetrics.audio` instead."
    )
    return scale_invariant_signal_noise_ratio(preds=preds, target=target)


def _signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    _future_warning(
        "Importing as `from torchmetrics.functional import signal_noise_ratio` was deprecated"
        " and will be removed in 2.0; use importing from domain specific sub-package `torchmetrics.audio` instead."
    )
    return signal_noise_ratio(preds=preds, target=target, zero_mean=zero_mean)
