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
from typing import Any, Tuple, Union

import torch
from torch import Tensor, nn

from torchmetrics.metric import Metric


class MetricTracker(nn.ModuleList):
    """A wrapper class that can help keeping track of a metric over time and implement useful methods. The wrapper
    implements the standard `update`, `compute`, `reset` methods that just calls corresponding method of the
    currently tracked metric. However, the following additional methods are provided:

        -``MetricTracker.n_steps``: number of metrics being tracked

        -``MetricTracker.increment()``: initialize a new metric for being tracked

        -``MetricTracker.compute_all()``: get the metric value for all steps

        -``MetricTracker.best_metric()``: returns the best value

    Args:
        metric: instance of a torchmetric modular to keep track of at each timestep.
        maximize: bool indicating if higher metric values are better (`True`) or lower
            is better (`False`)

    Example:

        >>> from torchmetrics import Accuracy, MetricTracker
        >>> _ = torch.manual_seed(42)
        >>> tracker = MetricTracker(Accuracy(num_classes=10))
        >>> for epoch in range(5):
        ...     tracker.increment()
        ...     for batch_idx in range(5):
        ...         preds, target = torch.randint(10, (100,)), torch.randint(10, (100,))
        ...         tracker.update(preds, target)
        ...     print(f"current acc={tracker.compute()}")  # doctest: +NORMALIZE_WHITESPACE
        current acc=0.1120000034570694
        current acc=0.08799999952316284
        current acc=0.12600000202655792
        current acc=0.07999999821186066
        current acc=0.10199999809265137
        >>> best_acc, which_epoch = tracker.best_metric(return_step=True)
        >>> tracker.compute_all()
        tensor([0.1120, 0.0880, 0.1260, 0.0800, 0.1020])
    """

    def __init__(self, metric: Metric, maximize: bool = True) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise TypeError("metric arg need to be an instance of a torchmetrics metric" f" but got {metric}")
        self._base_metric = metric
        self.maximize = maximize

        self._increment_called = False

    @property
    def n_steps(self) -> int:
        """Returns the number of times the tracker has been incremented."""
        return len(self) - 1  # subtract the base metric

    def increment(self) -> None:
        """Creates a new instace of the input metric that will be updated next."""
        self._increment_called = True
        self.append(deepcopy(self._base_metric))

    def forward(self, *args, **kwargs) -> None:  # type: ignore
        """Calls forward of the current metric being tracked."""
        self._check_for_increment("forward")
        return self[-1](*args, **kwargs)

    def update(self, *args, **kwargs) -> None:  # type: ignore
        """Updates the current metric being tracked."""
        self._check_for_increment("update")
        self[-1].update(*args, **kwargs)

    def compute(self) -> Any:
        """Call compute of the current metric being tracked."""
        self._check_for_increment("compute")
        return self[-1].compute()

    def compute_all(self) -> Tensor:
        """Compute the metric value for all tracked metrics."""
        self._check_for_increment("compute_all")
        return torch.stack([metric.compute() for i, metric in enumerate(self) if i != 0], dim=0)

    def reset(self) -> None:
        """Resets the current metric being tracked."""
        self[-1].reset()

    def reset_all(self) -> None:
        """Resets all metrics being tracked."""
        for metric in self:
            metric.reset()

    def best_metric(self, return_step: bool = False) -> Union[float, Tuple[int, float]]:
        """Returns the highest metric out of all tracked.

        Args:
            return_step: If `True` will also return the step with the highest metric value.

        Returns:
            The best metric value, and optionally the timestep.
        """
        fn = torch.max if self.maximize else torch.min
        idx, max = fn(self.compute_all(), 0)
        if return_step:
            return idx.item(), max.item()
        return max.item()

    def _check_for_increment(self, method: str) -> None:
        if not self._increment_called:
            raise ValueError(f"`{method}` cannot be called before `.increment()` has been called")
