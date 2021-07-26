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
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, nn

from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_7


def _bootstrap_sampler(
    size: int,
    sampling_strategy: str = 'poisson',
) -> Tensor:
    """ Resample a tensor along its first dimension with replacement
    Args:
        size: number of samples
        sampling_strategy: the strategy to use for sampling, either ``'poisson'`` or ``'multinomial'``
        generator: a instance of ``torch.Generator`` that controls the sampling

    Returns:
        resampled tensor

    """
    if sampling_strategy == 'poisson':
        p = torch.distributions.Poisson(1)
        n = p.sample((size, ))
        return torch.arange(size).repeat_interleave(n.long(), dim=0)
    if sampling_strategy == 'multinomial':
        idx = torch.multinomial(torch.ones(size), num_samples=size, replacement=True)
        return idx
    raise ValueError('Unknown sampling strategy')


class BootStrapper(Metric):

    def __init__(
        self,
        base_metric: Metric,
        num_bootstraps: int = 10,
        mean: bool = True,
        std: bool = True,
        quantile: Optional[Union[float, Tensor]] = None,
        raw: bool = False,
        sampling_strategy: str = 'poisson',
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ) -> None:
        r"""
        Use to turn a metric into a `bootstrapped <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_
        metric that can automate the process of getting confidence intervals for metric values. This wrapper
        class basically keeps multiple copies of the same base metric in memory and whenever ``update`` or
        ``forward`` is called, all input tensors are resampled (with replacement) along the first dimension.

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
            sampling_strategy:
                Determines how to produce bootstrapped samplings. Either ``'poisson'`` or ``multinomial``.
                If ``'possion'`` is chosen, the number of times each sample will be included in the bootstrap
                will be given by :math:`n\sim Poisson(\lambda=1)`, which approximates the true bootstrap distribution
                when the number of samples is large. If ``'multinomial'`` is chosen, we will apply true bootstrapping
                at the batch level to approximate bootstrapping over the hole dataset.
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
            >>> from pprint import pprint
            >>> from torchmetrics import Accuracy, BootStrapper
            >>> _ = torch.manual_seed(123)
            >>> base_metric = Accuracy()
            >>> bootstrap = BootStrapper(base_metric, num_bootstraps=20)
            >>> bootstrap.update(torch.randint(5, (20,)), torch.randint(5, (20,)))
            >>> output = bootstrap.compute()
            >>> pprint(output)
            {'mean': tensor(0.2205), 'std': tensor(0.0859)}

        """
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        if not isinstance(base_metric, Metric):
            raise ValueError(
                "Expected base metric to be an instance of torchmetrics.Metric"
                f" but received {base_metric}"
            )

        self.metrics = nn.ModuleList([deepcopy(base_metric) for _ in range(num_bootstraps)])
        self.num_bootstraps = num_bootstraps

        self.mean = mean
        self.std = std
        if quantile is not None and not _TORCH_GREATER_EQUAL_1_7:
            raise ValueError('quantile argument can only be used with pytorch v1.7 or higher')
        self.quantile = quantile
        self.raw = raw

        allowed_sampling = ('poisson', 'multinomial')
        if sampling_strategy not in allowed_sampling:
            raise ValueError(
                f"Expected argument ``sampling_strategy`` to be one of {allowed_sampling}"
                f" but recieved {sampling_strategy}"
            )
        self.sampling_strategy = sampling_strategy

    def update(self, *args: Any, **kwargs: Any) -> None:
        """ Updates the state of the base metric. Any tensor passed in will be bootstrapped along dimension 0 """
        for idx in range(self.num_bootstraps):
            args_sizes = apply_to_collection(args, Tensor, len)
            kwargs_sizes = list(apply_to_collection(kwargs, Tensor, len))
            if len(args_sizes) > 0:
                size = args_sizes[0]
            elif len(kwargs_sizes) > 0:
                size = kwargs_sizes[0]
            else:
                raise ValueError('None of the input contained tensors, so could not determine the sampling size')
            sample_idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy)
            new_args = apply_to_collection(args, Tensor, torch.index_select, dim=0, index=sample_idx)
            new_kwargs = apply_to_collection(kwargs, Tensor, torch.index_select, dim=0, index=sample_idx)
            self.metrics[idx].update(*new_args, **new_kwargs)

    def compute(self) -> Dict[str, Tensor]:
        """ Computes the bootstrapped metric values. Allways returns a dict of tensors, which can contain the
        following keys: ``mean``, ``std``, ``quantile`` and ``raw`` depending on how the class was initialized
        """
        computed_vals = torch.stack([m.compute() for m in self.metrics], dim=0)
        output_dict = {}
        if self.mean:
            output_dict['mean'] = computed_vals.mean(dim=0)
        if self.std:
            output_dict['std'] = computed_vals.std(dim=0)
        if self.quantile is not None:
            output_dict['quantile'] = torch.quantile(computed_vals, self.quantile)
        if self.raw:
            output_dict['raw'] = computed_vals
        return output_dict
