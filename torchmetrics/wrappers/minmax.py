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
from torchmetrics.metric import Metric
from typing import Any

class MinMaxMetric(Metric):
    """Wrapper Metric that tracks both the minimum and maximum of a scalar/tensor across an experiment."""

    def __init__(self, base_metric: Metric, dist_sync_on_step:bool=False, min_bound_init:float=1., max_bound_init:float=0.):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._base_metric = base_metric
        self.add_state("min_val", default=torch.tensor(min_bound_init))
        self.add_state("max_val", default=torch.tensor(max_bound_init))
        self.min_bound_init = min_bound_init
        self.max_bound_init = max_bound_init

    def update(self, *args: Any, **kwargs: Any):
        "Update underlying metric"
        self._base_metric.update(*args, **kwargs)
        

    def compute(self):
        "Compute underlying metric as well as max and min values."
        val = self._base_metric.compute()
        self.max_val = val if self.max_val < val else self.max_val
        self.min_val = val if self.min_val > val else self.min_val
        return {"raw" : val, "max" : self.max_val, "min" : self.min_val}

    def reset(self):
        "Sets max_val and min_val to 0. and resets the base metric."
        self.max_val = self.max_bound_init
        self.min_val = self.min_bound_init
        self._base_metric.reset()


