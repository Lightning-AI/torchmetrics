from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _weighted_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Weighted Absolute Percentage Error. Checks for same shape
    of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        epsilon: Avoids ZeroDivisionError.
    """

    _check_same_shape(preds, target)

    sum_abs_error = (preds - target).abs().sum()
    sum_scale = target.abs().sum()

    return sum_abs_error, sum_scale


def _weighted_absolute_percentage_error_compute(
    sum_abs_error: Tensor,
    sum_scale: Tensor,
    epsilon: float = 1.17e-06,
) -> Tensor:
    """Computes Weighted Absolute Percentage Error.

    Args:
        num_obs: Number of predictions or observations
    """

    return sum_abs_error / torch.clamp(sum_scale, min=epsilon)


def weighted_absolute_percentage_error(preds: Tensor, target: Tensor) -> Tensor:
    r"""
    Computes weighted absolute percentage error (WAPE_):

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with WAPE.

    Example:


    """
    sum_abs_error, sum_scale = _weighted_absolute_percentage_error_update(
        preds,
        target,
    )
    weighted_ape = _weighted_absolute_percentage_error_compute(
        sum_abs_error,
        sum_scale,
    )

    return weighted_ape
