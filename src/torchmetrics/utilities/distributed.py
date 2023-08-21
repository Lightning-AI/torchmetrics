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
from typing import Any, List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import Dataset
from typing_extensions import Literal


def reduce(x: Tensor, reduction: Literal["elementwise_mean", "sum", "none", None]) -> Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        x: the tensor, which shall be reduced
        reduction:  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given

    """
    if reduction == "elementwise_mean":
        return torch.mean(x)
    if reduction == "none" or reduction is None:
        return x
    if reduction == "sum":
        return torch.sum(x)
    raise ValueError("Reduction parameter unknown.")


def class_reduce(
    num: Tensor,
    denom: Tensor,
    weights: Tensor,
    class_reduction: Literal["micro", "macro", "weighted", "none", None] = "none",
) -> Tensor:
    """Reduce classification metrics of the form ``num / denom * weights``.

    For example for calculating standard accuracy the num would be number of true positives per class, denom would be
    the support per class, and weights would be a tensor of 1s.

    Args:
        num: numerator tensor
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems:

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'`` or ``None``: returns calculated metric per class

    Raises:
        ValueError:
            If ``class_reduction`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"`` or ``None``.

    """
    valid_reduction = ("micro", "macro", "weighted", "none", None)
    fraction = torch.sum(num) / torch.sum(denom) if class_reduction == "micro" else num / denom

    # We need to take care of instances where the denom can be 0
    # for some (or all) classes which will produce nans
    fraction[fraction != fraction] = 0

    if class_reduction == "micro":
        return fraction
    if class_reduction == "macro":
        return torch.mean(fraction)
    if class_reduction == "weighted":
        return torch.sum(fraction * (weights.float() / torch.sum(weights)))
    if class_reduction == "none" or class_reduction is None:
        return fraction

    raise ValueError(f"Reduction parameter {class_reduction} unknown. Choose between one of these: {valid_reduction}")


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            ``gathered_result[i]`` corresponds to result tensor from process ``i``

    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


class EvaluationDistributedSampler(torch.utils.data.DistributedSampler):
    """A specialized distributed sampler for evaluation (test and validation).

    It is derived from the PyTorch DistributedSampler, with one core difference: it doesn't add extra samples to make
    the data evenly divisible across devices. This is important while evaluating, as adding extra samples will screw
    the results towards those duplicated samples.

    Normally not adding the extra samples would lead to processes becoming out of sync, but this is handled by the
    custom syncronization in Torchmetrics. Thus this sampler does not in general secure that distributed operations
    are working outside of Torchmetrics.

    Arguments are the same as DistributedSampler, and this implementation only overrides the __init__ method.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in distributed training. By default,
            :attr:`world_size` is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`. By default, :attr:`rank` is
            retrieved from the current distributed group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the indices.
        seed (int, optional): random seed used to shuffle the sampler if :attr:`shuffle=True`. This number should be
            identical across all processes in the distributed group.
        drop_last (bool, optional): if ``True``, then the sampler will drop the tail of the data to make it evenly
            divisible across the number of replicas.

    For a full example on how to use this sampler, using both bare PyTorch but also PyTorch Lightning,
    check out the `distributed_evaluation.py` file in the examples folder.

    Example::
        The distributed sampler is always intended to be used in conjunction with a DataLoader:

        >>> import torch
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from torchmetrics.utilities.distributed import EvaluationDistributedSampler
        >>> dataset = TensorDataset(torch.arange(10))
        >>> dataloader = DataLoader(
        ...   dataset, sampler=EvaluationDistributedSampler(dataset, num_replicas=2)
        ... )  # doctest: +SKIP

    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        # From:
        # https://github.com/pytorch/pytorch/issues/25162#issuecomment-1227647626
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)

        len_dataset = len(self.dataset)  # type: ignore[arg-type]
        if not self.drop_last and len_dataset % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            if self.rank >= len_dataset % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len_dataset
