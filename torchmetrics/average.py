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
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric


class AverageMeter(Metric):
    """Computes the average of a stream of values.

    Forward accepts
        - ``value`` (float tensor): ``(...)``
        - ``weight`` (float tensor): ``(...)``

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is
            set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
            default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state.
            When `None`, DDP will be used to perform the allgather.

    Example::
        >>> from torchmetrics import AverageMeter
        >>> avg = AverageMeter()
        >>> avg.update(3)
        >>> avg.update(1)
        >>> avg.compute()
        tensor(2.)

        >>> avg = AverageMeter()
        >>> values = torch.tensor([1., 2., 3.])
        >>> avg(values)
        tensor(2.)

        >>> avg = AverageMeter()
        >>> values = torch.tensor([1., 2.])
        >>> weights = torch.tensor([3., 1.])
        >>> avg(values, weights)
        tensor(1.2500)
    """

    value: Tensor
    weight: Tensor

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("value", torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("weight", torch.zeros(()), dist_reduce_fx="sum")

    # TODO: need to be strings because Unions are not pickleable in Python 3.6
    def update(self, value: "Union[Tensor, float]", weight: "Union[Tensor, float]" = 1.0) -> None:  # type: ignore
        """Updates the average with.

        Args:
            value: A tensor of observations (can also be a scalar value)
            weight: The weight of each observation (automatically broadcasted
                to fit ``value``)
        """
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.value.device)
        if not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32, device=self.weight.device)

        # braodcast_to only supported on PyTorch 1.8+
        if not hasattr(torch, "broadcast_to"):
            if weight.shape == ():
                weight = torch.ones_like(value) * weight
            if weight.shape != value.shape:
                raise ValueError("Broadcasting not supported on PyTorch <1.8")
        else:
            weight = torch.broadcast_to(weight, value.shape)

        self.value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        return self.value / self.weight
