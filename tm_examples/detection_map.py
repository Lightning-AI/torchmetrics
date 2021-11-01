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
"""An example of how to the predictions and target should be defined for the MAP object detection metric To run:
python detection_map.py."""

import torch

from torchmetrics import MAP

# Preds should be a list of elements, where each element is a dict
# containing 3 keys: boxes, scores, labels
preds = [
    dict(
        # The boxes keyword should contain an [N,4] tensor,
        # where N is the number of detected boxes and ...
        boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]),
        scores=torch.Tensor([0.536]),
        labels=torch.IntTensor([0]),
    )
]

target = [
    dict(
        boxes=torch.Tensor([[214.0, 41.0, 562.0, 285.0]]),
        labels=torch.IntTensor([0]),
    )
]

if __name__ == "__main__":
    metric = MAP()
    metric.update(preds, target)
    result = metric.compute()
    print(result)
