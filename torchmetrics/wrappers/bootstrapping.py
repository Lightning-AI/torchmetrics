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
from copy import deepcopy
from typing import Any, Callable, List, Optional, Union

import torch
from torch import nn
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_7


def _bootstrap_sampler(
    tensor: Tensor, size: Optional[int] = None, generator: Optional[torch.Generator] = None
) -> Tensor:
    """ Resample a tensor along its first dimension with replacement
    Args:
        tensor: tensor to resample
        size: number of samples in new tensor. Defauls to same size as input tensor
        generator: a instance of ``torch.Generator`` that controls the sampling

    Returns:
        resampled tensor

    """
    if size is None:
        size = tensor.shape[0]
    idx = torch.multinomial(
        torch.ones(tensor.shape[0], device=tensor.device),
        num_samples=size,
        replacement=True,
        generator=generator
    )
    return tensor[idx]


class BootStrapper(Metric):
    def __init__(
        self,
        base_metric: Metric,
        num_bootstraps: int = 10,
        mean: bool = True,
        std: bool = True,
        quantile: Optional[Union[float, Tensor]] = None,
        raw: bool = False,
        generator: Optional[torch.Generator] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ) -> None:
        """
        Use to turn a metric into a bootstrapped metric that can automate the process of getting confidence
        intervals for metric values. This wrapper class basically keeps multiple copies of the same base metric
        in memory and whenever ``update`` or ``forward`` is called, all input tensors are resampled
        (with replacement) along the first dimension.

        Args:
            base_metric:
                base metric class to wrap
            num_bootstraps:
                number of copies to make of the base metric for bootstrapping
            mean:
                if ``True`` return the mean of the bootstraps
            std:
                if ``True`` return the standard diviation of the bootstraps
            quantile:
                if given, returns the quantile of the bootstraps. Can only be used with
                pytorch version 1.6 or higher
            raw:
                if ``True``, return all bootstrapped values
            generator:
                A pytorch random number generator for the bootstrap sampler
            compute_on_step:
                Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
            dist_sync_on_step:
                Synchronize metric state across processes at each ``forward()``
                before returning the value at the step
            process_group:
                Specify the process group on which synchronization is called.
                default: ``None`` (which selects the entire world)
            dist_sync_fn:
                Callback that performs the allgather operation on the metric state. When ``None``, DDP
                will be used to perform the allgather.

        Example::
            >>> from torchmetrics.wrappers import BootStrapper
            >>> from torchmetrics import Accuracy
            >>> generator = torch.manual_seed(0)
            >>> base_metric = Accuracy()
            >>> bootstrap = BootStrapper(base_metric, num_bootstraps=20, generator=generator)
            >>> bootstrap.update(torch.randint(5, (20,)), torch.randint(5, (20,)))
            >>> output = bootstrap.compute()
            >>> mean, std = output
            >>> print(mean, std)
            tensor(0.2175) tensor(0.0950)

        """
        super().__init__(
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn
        )
        if not isinstance(base_metric, Metric):
            raise ValueError("Expected base metric to be an instance of torchmetrics.Metric"
                             f" but received {base_metric}")

        self.metrics = nn.ModuleList([deepcopy(base_metric) for _ in range(num_bootstraps)])
        self.num_bootstraps = num_bootstraps

        self.mean = mean
        self.std = std
        if quantile is not None and not _TORCH_GREATER_EQUAL_1_7:
            raise ValueError('quantile argument can only be used with pytorch v1.6 or higher')
        self.quantile = quantile
        self.raw = raw

        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError(
                "Expected argument ``generator`` to be an instance of ``torch.Generator``"
                f"but received {generator}"
            )
        self.generator = generator

    def update(self, *args: Any, **kwargs: Any) -> None:
        """ Updates the state of the base metric. Any tensor passed in will be bootstrapped
        along dimension 0
        """
        for idx in range(self.num_bootstraps):
            new_args = apply_to_collection(args, Tensor, _bootstrap_sampler, generator=self.generator)
            new_kwargs = apply_to_collection(kwargs, Tensor, _bootstrap_sampler, generator=self.generator)
            self.metrics[idx].update(*new_args, **new_kwargs)

    def compute(self) -> List[Tensor]:
        """ Computes the bootstrapped metric values. Allways returns a list of tensors, but the content of
        the list depends on how the class was initialized
        """
        computed_vals = torch.stack([m.compute() for m in self.metrics], dim=0)
        output = []
        if self.mean:
            output.append(computed_vals.mean(dim=0))
        if self.std:
            output.append(computed_vals.std(dim=0))
        if self.quantile is not None:
            output.append(torch.quantile(computed_vals, self.quantile))
        if self.raw:
            output.append(computed_vals)
        return output
