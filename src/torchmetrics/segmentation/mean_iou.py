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


class MeanIOU(Metric):
    """Computes Mean Intersection over Union (mIoU) for semantic segmentation."""

    def __init__(self) -> None:
        pass

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with the new data."""

    def compute(self) -> Tensor:
        """Update the state with the new data."""
