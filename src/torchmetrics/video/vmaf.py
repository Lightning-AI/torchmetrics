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
from torch import Tensor
from torchmetrics.metric import Metric
from vmaf_torch.vmaf import VMAF as VMAF_torch

class VMAF(Metric):
    """
    
    

    .. note::
        This implementation requires you to have vmaf-torch installed: https://github.com/alvitrioliks/VMAF-torch.
        Install either by cloning the repository and running `pip install .` or with `pip install torchmetrics[video]`.
    
    """
    def __init__(self):
        super().__init__()
        self.backend = VMAF_torch()
        self.backend.compile()

    def update(self, preds: Tensor, target: Tensor) -> None:
        result = self.backend(ref=target, dist=preds)
        self.backend.compute_adm_features(ref=target, dist=preds)
        self.backend.compute_vif_features(ref=target, dist=preds)
        self.backend.compute_motion(ref=target)

    def compute(self) -> Tensor:
        pass