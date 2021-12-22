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

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric


class MinMaxMetric(Metric):
    """Wrapper Metric that tracks both the minimum and maximum of a scalar/tensor across an experiment. The min/max
    value will be updated each time `.compute` is called.

    Args:
        base_metric:
            The metric of which you want to keep track of its maximum and minimum values.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Raises:
        ValueError
            If ``base_metric` argument is not an subclasses instance of ``torchmetrics.Metric``

    Example::
        >>> import torch
        >>> from torchmetrics import Accuracy
        >>> from pprint import pprint
        >>> base_metric = Accuracy()
        >>> minmax_metric = MinMaxMetric(base_metric)
        >>> preds_1 = torch.Tensor([[0.1, 0.9], [0.2, 0.8]])
        >>> preds_2 = torch.Tensor([[0.9, 0.1], [0.2, 0.8]])
        >>> labels = torch.Tensor([[0, 1], [0, 1]]).long()
        >>> pprint(minmax_metric(preds_1, labels))
        {'max': tensor(1.), 'min': tensor(1.), 'raw': tensor(1.)}
        >>> pprint(minmax_metric.compute())
        {'max': tensor(1.), 'min': tensor(1.), 'raw': tensor(1.)}
        >>> minmax_metric.update(preds_2, labels)
        >>> pprint(minmax_metric.compute())
        {'max': tensor(1.), 'min': tensor(0.7500), 'raw': tensor(0.7500)}
    """

    min_val: Tensor
    max_val: Tensor

    def __init__(
        self,
        base_metric: Metric,
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
        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected base metric to be an instance of `torchmetrics.Metric` but received {base_metric}"
            )
        self._base_metric = base_metric
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
        """Updates the underlying metric."""
        self._base_metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:  # type: ignore
        """Computes the underlying metric as well as max and min values for this metric.

        Returns a dictionary that consists of the computed value (``raw``), as well as the minimum (``min``) and maximum
        (``max``) values.
        """
        val = self._base_metric.compute()
        if not self._is_suitable_val(val):
            raise RuntimeError(
                f"Returned value from base metric should be a scalar (int, float or tensor of size 1, but got {val}"
            )
        self.max_val = val if self.max_val < val else self.max_val
        self.min_val = val if self.min_val > val else self.min_val
        return {"raw": val, "max": self.max_val, "min": self.min_val}

    def reset(self) -> None:
        """Sets ``max_val`` and ``min_val`` to the initialization bounds and resets the base metric."""
        super().reset()
        self._base_metric.reset()

    @staticmethod
    def _is_suitable_val(val: Union[int, float, Tensor]) -> bool:
        """Utility function that checks whether min/max value."""
        if isinstance(val, (int, float)):
            return True
        if isinstance(val, Tensor):
            return val.numel() == 1
        return False
