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

from typing import Any, Dict

import torch
from torch import Tensor

from torchmetrics.metric import Metric

def _is_suitable_val(val: Any) -> bool:
    """Utility function that checks whether min/max value is either a:
       - int
       - float
       - tensor with 1 element
    """
    print(val)
    print(type(val))
    if (type(val) == int) or (type(val) == float):
        return True
    elif type(val) == torch.Tensor:
        print(val.size())
        if val.size() == torch.Size([]):
            return True
        else:
            return False
    else:
        return False 



class MinMaxMetric(Metric):
    """Wrapper Metric that tracks both the minimum and maximum of a scalar/tensor across an experiment.

    Note:
        Make sure you pass proper initialization values to the ``min_bound_init`` and ``max_bound_init`` parameters.
        For the ``Accuracy`` metric, the defaults of ``0.0`` and ``1.0`` make sense,
        however, for other metrics you will likely want to use different initialization values.

    Args:
        base_metric:
            The metric of which you want to keep track of its maximum and minimum values.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        min_bound_init:
            Initialization value of the ``min`` parameter. default: 0.0
        max_bound_init:
            Initialization value of the ``max`` parameter. default: 1.0
    """

    def __init__(
        self,
        base_metric: Metric,
        dist_sync_on_step: bool = False,
        min_bound_init: float = 1.0,
        max_bound_init: float = 0.0,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._base_metric = base_metric
        self.add_state("min_val", default=torch.tensor(min_bound_init))
        self.add_state("max_val", default=torch.tensor(max_bound_init))
        self.min_bound_init = min_bound_init
        self.max_bound_init = max_bound_init

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
        """Updates the underlying metric."""
        self._base_metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:  # type: ignore
        """Computes the underlying metric as well as max and min values for this metric.

        Returns a dictionary that consists of the computed value (``raw``), as well as the minimum (``min``) and maximum
        (``max``) values.
        """
        val = self._base_metric.compute()
        isv = _is_suitable_val(val)
        print(isv)
        assert _is_suitable_val(val), "Computed Base Metric should be a scalar (Int, Float or Tensor of Size 1)"
        self.max_val = val if self.max_val < val else self.max_val
        self.min_val = val if self.min_val > val else self.min_val
        return {"raw": val, "max": self.max_val, "min": self.min_val}

    def reset(self) -> None:
        """Sets ``max_val`` and ``min_val`` to the initialization bounds and resets the base metric."""
        self.max_val = self.max_bound_init
        self.min_val = self.min_bound_init
        self._base_metric.reset()
