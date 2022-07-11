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
from itertools import permutations
from typing import Any, Callable, Tuple, Union
from warnings import warn

import torch
from torch import Tensor

from torchmetrics.utilities.imports import _SCIPY_AVAILABLE

# _ps_dict: cache of permutations
# it's necessary to cache it, otherwise it will consume a large amount of time
_ps_dict: dict = {}  # _ps_dict[str(spk_num)+str(device)] = permutations


def _find_best_perm_by_linear_sum_assignment(
    metric_mtx: Tensor,
    eval_func: Union[torch.min, torch.max],
) -> Tuple[Tensor, Tensor]:
    """Solves the linear sum assignment problem using scipy, and returns the best metric values and the
    corresponding permutations.

    Args:
        metric_mtx: the metric matrix, shape [batch_size, spk_num, spk_num]
        eval_func: the function to reduce the metric values of different the permutations

    Returns:
        best_metric: shape ``[batch]``
        best_perm: shape ``[batch, spk]``
    """
    from scipy.optimize import linear_sum_assignment

    mmtx = metric_mtx.detach().cpu()
    best_perm = torch.tensor([linear_sum_assignment(pwm, eval_func == torch.max)[1] for pwm in mmtx])
    best_perm = best_perm.to(metric_mtx.device)
    best_metric = torch.gather(metric_mtx, 2, best_perm[:, :, None]).mean([-1, -2])
    return best_metric, best_perm  # shape [batch], shape [batch, spk]


def _find_best_perm_by_exhaustive_method(
    metric_mtx: Tensor,
    eval_func: Union[torch.min, torch.max],
) -> Tuple[Tensor, Tensor]:
    """Solves the linear sum assignment problem using exhaustive method, i.e. exhaustively calculates the metric
    values of all possible permutations, and returns the best metric values and the corresponding permutations.

    Args:
        metric_mtx: the metric matrix, shape ``[batch_size, spk_num, spk_num]``
        eval_func: the function to reduce the metric values of different the permutations

    Returns:
        best_metric: shape ``[batch]``
        best_perm: shape ``[batch, spk]``
    """
    # create/read/cache the permutations and its indexes
    # reading from cache would be much faster than creating in CPU then moving to GPU
    batch_size, spk_num = metric_mtx.shape[:2]
    key = str(spk_num) + str(metric_mtx.device)
    if key not in _ps_dict:
        # ps: all the permutations, shape [spk_num, perm_num]
        # ps: In i-th permutation, the predcition corresponds to the j-th target is ps[j,i]
        ps = torch.tensor(list(permutations(range(spk_num))), device=metric_mtx.device).T
        _ps_dict[key] = ps
    else:
        ps = _ps_dict[key]  # all the permutations, shape [spk_num, perm_num]

    # find the metric of each permutation
    perm_num = ps.shape[-1]
    # shape [batch_size, spk_num, perm_num]
    bps = ps[None, ...].expand(batch_size, spk_num, perm_num)
    # shape [batch_size, spk_num, perm_num]
    metric_of_ps_details = torch.gather(metric_mtx, 2, bps)
    # shape [batch_size, perm_num]
    metric_of_ps = metric_of_ps_details.mean(dim=1)

    # find the best metric and best permutation
    best_metric, best_indexes = eval_func(metric_of_ps, dim=1)
    best_indexes = best_indexes.detach()
    best_perm = ps.T[best_indexes, :]
    return best_metric, best_perm  # shape [batch], shape [batch, spk]


def permutation_invariant_training(
    preds: Tensor, target: Tensor, metric_func: Callable, eval_func: str = "max", **kwargs: Any
) -> Tuple[Tensor, Tensor]:
    """Permutation invariant training (PIT). The ``permutation_invariant_training`` implements the famous
    Permutation Invariant Training method.

    [1] in speech separation field in order to calculate audio metrics in a permutation invariant way.

    Args:
        preds: shape ``[batch, spk, ...]``
        target: shape ``[batch, spk, ...]``
        metric_func: a metric function accept a batch of target and estimate,
            i.e. ``metric_func(preds[:, i, ...], target[:, j, ...])``, and returns a batch of metric tensors ``[batch]``
        eval_func: the function to find the best permutation, can be ``'min'`` or ``'max'``,
            i.e. the smaller the better or the larger the better.
        kwargs: Additional args for metric_func

    Returns:
        best_metric of shape ``[batch]``
        best_perm of shape ``[batch]``

    Example:
        >>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        >>> # [batch, spk, time]
        >>> preds = torch.tensor([[[-0.0579,  0.3560, -0.9604], [-0.1719,  0.3205,  0.2951]]])
        >>> target = torch.tensor([[[ 1.0958, -0.1648,  0.5228], [-0.4100,  1.1942, -0.5103]]])
        >>> best_metric, best_perm = permutation_invariant_training(
        ...     preds, target, scale_invariant_signal_distortion_ratio, 'max')
        >>> best_metric
        tensor([-5.1091])
        >>> best_perm
        tensor([[0, 1]])
        >>> pit_permutate(preds, best_perm)
        tensor([[[-0.0579,  0.3560, -0.9604],
                 [-0.1719,  0.3205,  0.2951]]])

    Reference:
        [1]	`Permutation Invariant Training of Deep Models`_
    """
    if preds.shape[0:2] != target.shape[0:2]:
        raise RuntimeError(
            "Predictions and targets are expected to have the same shape at the batch and speaker dimensions"
        )
    if eval_func not in ["max", "min"]:
        raise ValueError(f'eval_func can only be "max" or "min" but got {eval_func}')
    if target.ndim < 2:
        raise ValueError(f"Inputs must be of shape [batch, spk, ...], got {target.shape} and {preds.shape} instead")

    # calculate the metric matrix
    batch_size, spk_num = target.shape[0:2]
    metric_mtx = None
    for target_idx in range(spk_num):  # we have spk_num speeches in target in each sample
        for preds_idx in range(spk_num):  # we have spk_num speeches in preds in each sample
            if metric_mtx is not None:
                metric_mtx[:, target_idx, preds_idx] = metric_func(
                    preds[:, preds_idx, ...], target[:, target_idx, ...], **kwargs
                )
            else:
                first_ele = metric_func(preds[:, preds_idx, ...], target[:, target_idx, ...], **kwargs)
                metric_mtx = torch.empty((batch_size, spk_num, spk_num), dtype=first_ele.dtype, device=first_ele.device)
                metric_mtx[:, target_idx, preds_idx] = first_ele

    # find best
    op = torch.max if eval_func == "max" else torch.min
    if spk_num < 3 or not _SCIPY_AVAILABLE:
        if spk_num >= 3 and not _SCIPY_AVAILABLE:
            warn(f"In pit metric for speaker-num {spk_num}>3, we recommend installing scipy for better performance")

        best_metric, best_perm = _find_best_perm_by_exhaustive_method(metric_mtx, op)
    else:
        best_metric, best_perm = _find_best_perm_by_linear_sum_assignment(metric_mtx, op)

    return best_metric, best_perm


def pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """Permutate estimate according to perm.

    Args:
        preds: the estimates you want to permutate, shape [batch, spk, ...]
        perm: the permutation returned from permutation_invariant_training, shape [batch, spk]

    Returns:
        Tensor: the permutated version of estimate
    """
    preds_pmted = torch.stack([torch.index_select(pred, 0, p) for pred, p in zip(preds, perm)])
    return preds_pmted
