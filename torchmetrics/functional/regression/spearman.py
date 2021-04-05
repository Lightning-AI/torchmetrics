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
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape


def _find_repeats(data: Tensor):
    """ find and return values which have repeats i.e. the same value are more than once in the tensor """
    temp = data.detach().clone()
    temp = temp.sort()[0]
    
    change = torch.cat([torch.tensor([True]), temp[1:] != temp[:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]])]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]


def _rank_data(data: Tensor):
    """ Calculate the rank for each element of a tensor. The rank refers to the indices of an element in the
    corresponding sorted tensor (starting from 1). Duplicates of the same value will be assigned the mean of
    their rank 
    
    Adopted from:
        https://github.com/scipy/scipy/blob/v1.6.2/scipy/stats/stats.py#L4140-L4303
    """
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n+1, dtype=torch.float)
    
    repeats = _find_repeats(data)
    for r in repeats:
        condition = (data == r).filled(False)
        rank[condition] = rank[condition].mean()
    return rank

def _spearman_corrcoef_update(preds: Tensor, target: Tensor):
    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got pred: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    
    if preds.ndim > 1 or target.ndim > 1:
        raise ValueError('Expected both predictions and target to be 1 dimensional tensors.')
    
    return preds, target
    
def _spearman_corrcoef_compute(preds: Tensor, target: Tensor):
    rank_preds = _rank_data(preds)
    rank_target = _rank_data(target)
    
    cov = ((rank_preds - rank_preds.mean()) * (rank_target - rank_target.mean())).sum()
    return cov / (rank_preds.std() * rank_target.std())
    
    
def spearman_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """
    
    """
    preds, target = _spearman_corrcoef_update(preds, target)
    return _spearman_corrcoef_compute(preds, target)
    
    