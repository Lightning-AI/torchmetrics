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
from typing import Any, Dict, Literal, Optional

from torch import Tensor, nn
from torch_fidelity.utils import create_sample_similarity

from torchmetrics.functional.image.perceptual_path_length import (
    _perceptual_path_length_validate_arguments,
    _validate_generator_model,
    perceptual_path_length,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


class PerceptualPathLength(Metric):
    """Computes the perceptual path length (PPL) of a generator."""

    def __init__(
        self,
        num_samples: int = 10_000,
        conditional: bool = False,
        batch_size: int = 128,
        interpolation_method: Literal["lerp", "slerp_any", "slerp_unit"] = "lerp",
        epsilon: float = 1e-4,
        resize: Optional[int] = 64,
        lower_discard: Optional[float] = 0.01,
        upper_discard: Optional[float] = 0.99,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _TORCH_FIDELITY_AVAILABLE:
            raise ModuleNotFoundError(
                "Metric `PerceptualPathLength` requires Torch Fidelity which is not installed."
                "Install with `pip install torch-fidelity` or `pip install torchmetrics[image]`"
            )
        _perceptual_path_length_validate_arguments(
            num_samples, conditional, batch_size, interpolation_method, epsilon, resize, lower_discard, upper_discard
        )
        self.num_samples = num_samples
        self.conditional = conditional
        self.batch_size = batch_size
        self.interpolation_method = interpolation_method
        self.epsilon = epsilon
        self.resize = resize
        self.lower_discard = lower_discard
        self.upper_discard = upper_discard

        self.sim_net = create_sample_similarity(
            "lpips-vgg16", sample_similarity_resize=resize, cuda=self.device == "cuda", verbose=False
        )

    def update(self, generator: nn.Module) -> None:
        """Update the generator model."""
        _validate_generator_model(generator, self.conditional)
        self.generator = generator

    def compute(self) -> Dict[str, Tensor]:
        """Compute the perceptual path length."""
        return perceptual_path_length(
            generator=self.generator,
            num_samples=self.num_samples,
            conditional=self.conditional,
            interpolation_method=self.interpolation_method,
            epsilon=self.epsilon,
            resize=self.resize,
            lower_discard=self.lower_discard,
            uppper_discard=self.upper_discard,
            sim_net=self.sim_net,
            device=self.device,
        )
