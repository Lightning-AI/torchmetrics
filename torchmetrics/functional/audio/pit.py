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
from torchmetrics.utilities.checks import _check_same_shape

from itertools import permutations
from typing import Callable, List, Tuple, Union
import torch
from torch.tensor import Tensor
from scipy.optimize import linear_sum_assignment

_ps_dict: dict = {}  # cache
_ps_idx_dict: dict = {}  # cache


def _find_best_perm_by_linear_sum_assignment(metric_mtx: torch.Tensor, eval_func: Union[torch.min, torch.max]):
    mmtx = metric_mtx.detach().cpu()
    best_perm = torch.tensor([linear_sum_assignment(pwm, eval_func == torch.max)[1] for pwm in mmtx]).to(metric_mtx.device)
    best_metric = torch.gather(metric_mtx, 2, best_perm[:, :, None]).mean([-1, -2])
    return best_metric, best_perm  # shape [batch], shape [batch, spk]


def _find_best_perm_by_exhuastive_method(metric_mtx: torch.Tensor, eval_func: Union[torch.min, torch.max]):
    # create/read/cache the permutations and its indexes
    # reading from cache would be much faster than creating in CPU then moving to GPU
    batch_size, spk_num = metric_mtx.shape[:2]
    key = str(spk_num) + str(metric_mtx.device)
    if key not in _ps_dict:
        # all the permutations, shape [perm_num, spk_num]
        ps = torch.tensor(list(permutations(range(spk_num))), device=metric_mtx.device)
        # shape [perm_num * spk_num]
        inc = torch.arange(0, spk_num * spk_num, step=spk_num, device=metric_mtx.device, dtype=ps.dtype).repeat(ps.shape[0])
        # the indexes for all permutations, shape [perm_num*spk_num]
        ps_idx = ps.view(-1) + inc
        # cache ps and ps_idx
        _ps_idx_dict[key] = ps_idx
        _ps_dict[key] = ps
    else:
        ps_idx = _ps_idx_dict[key]  # the indexes for all permutations, shape [perm_num*spk_num]
        ps = _ps_dict[key]  # all the permutations, shape [perm_num, spk_num]

    # find the metric of each permutation
    metric_of_ps_details = metric_mtx.view(batch_size, -1)[:, ps_idx].reshape(batch_size, -1, spk_num)  # shape [batch_size, perm_num, spk_num]
    metric_of_ps = metric_of_ps_details.mean(dim=2)  # shape [batch_size, perm_num]

    # find the best metric and best permutation
    best_metric, best_indexes = eval_func(metric_of_ps, dim=1)
    best_indexes = best_indexes.detach()
    best_perm = ps[best_indexes, :]
    return best_metric, best_perm  # shape [batch], shape [batch, spk]


def pit(preds: torch.Tensor, target: torch.Tensor, metric_func: Callable, eval_func: str = 'max', **kwargs) -> Tuple[Tensor, Tensor]:
    """ Permutation invariant training metric

    Args:
        target:
            shape [batch, spk, ...]
        preds:
            shape [batch, spk, ...]
        metric_func:
            a metric function accept a batch of target and estimate, i.e. metric_func(target[:, i, ...], estimate[:, j, ...]), and returns a batch of metric tensors [batch]
        eval_func:
            the function to find the best permutation, can be 'min' or 'max', i.e. the smaller the better or the larger the better.
        kwargs:
            additional args for metric_func

    Returns:
        best_metric of shape [batch], 
        best_perm of shape [batch]
    
    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import si_snr, pit, permutate
        >>> preds = torch.randn(3, 2, 5) # [batch, spk, time]
        >>> target = torch.randn(3, 2, 5) # [batch, spk, time]
        >>> best_metric, best_perm = pit(preds, target, si_snr, 'max')
        >>> best_metric
        tensor([-29.3482, -11.2862,  -9.2508])
        >>> best_perm
        tensor([[0, 1],
                [1, 0],
                [0, 1]])
        >>> preds_pmted = permutate(preds, best_perm)

    Reference:
        [1]	D. Yu, M. Kolbaek, Z.-H. Tan, J. Jensen, Permutation invariant training of deep models for speaker-independent multi-talker speech separation, in: 2017 IEEE Int. Conf. Acoust. Speech Signal Process. ICASSP, IEEE, New Orleans, LA, 2017: pp. 241–245. https://doi.org/10.1109/ICASSP.2017.7952154.
    """
    _check_same_shape(preds, target)
    if eval_func not in ['max', 'min']:
        raise RuntimeError('eval_func can only be "max" or "min"')
    if len(target.shape) < 2:
        raise RuntimeError(f"Inputs must be of shape [batch, spk, ...], got {target.shape} and {preds.shape} instead")

    # calculate the metric matrix
    batch_size, spk_num = target.shape[0:2]
    metric_mtx = torch.empty((batch_size, spk_num, spk_num), dtype=preds.dtype, device=target.device)
    for t in range(spk_num):
        for e in range(spk_num):
            metric_mtx[:, t, e] = metric_func(preds[:, e, ...], target[:, t, ...], **kwargs)

    # find best
    if spk_num < 3:
        best_metric, best_perm = _find_best_perm_by_exhuastive_method(metric_mtx, torch.max if eval_func == 'max' else torch.min)
    else:
        best_metric, best_perm = _find_best_perm_by_linear_sum_assignment(metric_mtx, torch.max if eval_func == 'max' else torch.min)

    return best_metric, best_perm


def permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """ permutate estimate according to perm

    Args:
        preds (Tensor): the estimates you want to permutate, shape [batch, spk, ...]
        perm (Tensor): the permutation returned from pit, shape [batch, spk]

    Returns:
        Tensor: the permutated version of estimate
    
    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import si_snr, pit, permutate
        >>> preds = torch.randn(3, 2, 5) # [batch, spk, time]
        >>> target = torch.randn(3, 2, 5) # [batch, spk, time]
        >>> best_metric, best_perm = pit(preds, target, si_snr, 'max')
        >>> preds_pmted = permutate(preds, best_perm)
    """
    preds_pmted = torch.stack([torch.index_select(pred, 0, p) for pred, p in zip(preds, perm)])
    return preds_pmted
