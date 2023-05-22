from typing import Any, Callable, Optional, Tuple

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.utilities.prints import _deprecated_root_import_func


def _permutation_invariant_training(
    preds: Tensor,
    target: Tensor,
    metric_func: Callable,
    mode: Literal["speaker-wise", "permutation-wise"] = "speaker-wise",
    eval_func: Literal["max", "min"] = "max",
    **kwargs: Any
) -> Tuple[Tensor, Tensor]:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([[[-0.0579,  0.3560, -0.9604], [-0.1719,  0.3205,  0.2951]]])
    >>> target = tensor([[[ 1.0958, -0.1648,  0.5228], [-0.4100,  1.1942, -0.5103]]])
    >>> best_metric, best_perm = _permutation_invariant_training(
    ...     preds, target, _scale_invariant_signal_distortion_ratio)
    >>> best_metric
    tensor([-5.1091])
    >>> best_perm
    tensor([[0, 1]])
    >>> pit_permutate(preds, best_perm)
    tensor([[[-0.0579,  0.3560, -0.9604],
             [-0.1719,  0.3205,  0.2951]]])
    """
    _deprecated_root_import_func("permutation_invariant_training", "audio")
    return permutation_invariant_training(
        preds=preds, target=target, metric_func=metric_func, mode=mode, eval_func=eval_func, **kwargs
    )


def _pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """Wrapper for deprecated import."""
    _deprecated_root_import_func("pit_permutate", "audio")
    return pit_permutate(preds=preds, perm=perm)


def _scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> _scale_invariant_signal_distortion_ratio(preds, target)
    tensor(18.4030)
    """
    _deprecated_root_import_func("scale_invariant_signal_distortion_ratio", "audio")
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=zero_mean)


def _signal_distortion_ratio(
    preds: Tensor,
    target: Tensor,
    use_cg_iter: Optional[int] = None,
    filter_length: int = 512,
    zero_mean: bool = False,
    load_diag: Optional[float] = None,
) -> Tensor:
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(1)
    >>> preds = torch.randn(8000)
    >>> target = torch.randn(8000)
    >>> _signal_distortion_ratio(preds, target)
    tensor(-12.0589)
    >>> # use with permutation_invariant_training
    >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
    >>> target = torch.randn(4, 2, 8000)
    >>> best_metric, best_perm = _permutation_invariant_training(preds, target, _signal_distortion_ratio)
    >>> best_metric
    tensor([-11.6375, -11.4358, -11.7148, -11.6325])
    >>> best_perm
    tensor([[1, 0],
            [0, 1],
            [1, 0],
            [0, 1]])
    """
    _deprecated_root_import_func("signal_distortion_ratio", "audio")
    return signal_distortion_ratio(
        preds=preds,
        target=target,
        use_cg_iter=use_cg_iter,
        filter_length=filter_length,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )


def _scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> _scale_invariant_signal_noise_ratio(preds, target)
    tensor(15.0918)
    """
    _deprecated_root_import_func("scale_invariant_signal_noise_ratio", "audio")
    return scale_invariant_signal_noise_ratio(preds=preds, target=target)


def _signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> _signal_noise_ratio(preds, target)
    tensor(16.1805)
    """
    _deprecated_root_import_func("signal_noise_ratio", "audio")
    return signal_noise_ratio(preds=preds, target=target, zero_mean=zero_mean)
