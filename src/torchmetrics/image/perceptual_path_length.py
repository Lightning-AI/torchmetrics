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
from typing import Any, Dict

from torch import Tensor, nn

from torchmetrics.functional.image.perceptual_path_length import perceptual_path_length
from torchmetrics.metric import Metric


class PerceptualPathLength(Metric):
    """Computes the perceptual path length (PPL) of a generator."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def update(self, generator: nn.Module) -> None:
        """Update the generator model."""
        self.generator = generator

    def compute(self) -> Dict[str, Tensor]:
        """Compute the perceptual path length."""
        return perceptual_path_length(self.generator)
