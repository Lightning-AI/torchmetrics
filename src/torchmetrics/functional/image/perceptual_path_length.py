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
import math
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_fidelity.noise import batch_lerp, batch_slerp_any, batch_slerp_unit
from torch_fidelity.utils import create_sample_similarity

from torchmetrics.functional.image.lpips import _get_net
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


def _validate_generator_model(generator: nn.Module, conditional: bool = False) -> None:
    """Validate that the user provided generator has the right methods and attributes."""
    if not hasattr(generator, "sample"):
        raise NotImplementedError(
            "The generator must have a `sample` method with signature `sample(num_samples: int) -> Tensor` where the"
            " returned tensor has shape `(num_samples, z_size)`."
        )
    if conditional and not hasattr(generator, "num_classes"):
        raise AttributeError("The generator must have a `num_classes` attribute when `conditional=True`.")


def _perceptual_path_length_validate_arguments(
    num_samples: int = 10_000,
    conditional: bool = False,
    batch_size: int = 128,
    interpolation_method: Literal["lerp", "slerp_any", "slerp_unit"] = "lerp",
    epsilon: float = 1e-4,
    resize: Optional[int] = 64,
    lower_discard: Optional[float] = 0.01,
    upper_discard: Optional[float] = 0.99,
) -> None:
    """Validate arguments for perceptual path length."""
    if not (isinstance(num_samples, int) and num_samples > 0):
        raise ValueError("Argument `num_samples` must be a positive integer, but got {num_samples}.")
    if not isinstance(conditional, bool):
        raise ValueError("Argument `conditional` must be a boolean, but got {conditional}.")
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError("Argument `batch_size` must be a positive integer, but got {batch_size}.")
    if interpolation_method not in ["lerp", "slerp_any", "slerp_unit"]:
        raise ValueError(
            f"Argument `interpolation_method` must be one of 'lerp', 'slerp_any', 'slerp_unit',"
            f"got {interpolation_method}."
        )
    if not (isinstance(epsilon, float) and epsilon > 0):
        raise ValueError("Argument `epsilon` must be a positive float, but got {epsilon}.")
    if resize is not None and not (isinstance(resize, int) and resize > 0):
        raise ValueError("Argument `resize` must be a positive integer or `None`, but got {resize}.")
    if lower_discard is not None and not (isinstance(lower_discard, float) and 0 <= lower_discard <= 1):
        raise ValueError("Argument `lower_discard` must be a float between 0 and 1 or `None`, but got {lower_discard}.")
    if upper_discard is not None and not (isinstance(upper_discard, float) and 0 <= upper_discard <= 1):
        raise ValueError("Argument `upper_discard` must be a float between 0 and 1 or `None`, but got {upper_discard}.")


def _interpolate(
    latents1: Tensor,
    latents2: Tensor,
    epsilon: float = 1e-4,
    interpolation_method: Literal["lerp", "slerp_any", "slerp_unit"] = "lerp",
) -> Tensor:
    if latents1.shape != latents2.shape:
        raise ValueError("Latents must have the same shape.")
    if interpolation_method == "lerp":
        return batch_lerp(latents1, latents2, epsilon)
    if interpolation_method == "slerp_any":
        return batch_slerp_unit(latents1, latents2, epsilon)
    if interpolation_method == "slerp_unit":
        return batch_slerp_any(latents1, latents2, epsilon)
    raise ValueError(
        f"Interpolation method {interpolation_method} not supported. Choose from 'lerp', 'slerp_any', 'slerp_unit'."
    )


def perceptual_path_length(
    generator: nn.Module,
    num_samples: int = 10_000,
    conditional: bool = False,
    batch_size: int = 128,
    interpolation_method: Literal["lerp", "slerp_any", "slerp_unit"] = "lerp",
    epsilon: float = 1e-4,
    resize: Optional[int] = 64,
    lower_discard: Optional[float] = 0.01,
    upper_discard: Optional[float] = 0.99,
    sim_net: Optional[nn.Module] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the perceptual path length (PPL) of a generator model."""
    if not _TORCH_FIDELITY_AVAILABLE:
        raise ModuleNotFoundError(
            "Metric `perceptual_path_length` requires Torch Fidelity which is not installed."
            "Install with `pip install torch-fidelity` or `pip install torchmetrics[image]`"
        )
    _perceptual_path_length_validate_arguments(
        num_samples, conditional, batch_size, interpolation_method, epsilon, lower_discard, upper_discard
    )
    _validate_generator_model(generator, conditional)
    generator = generator.to(device)

    latent1 = generator.sample(num_samples).to(device)
    latent2 = generator.sample(num_samples).to(device)
    latent2 = _interpolate(latent1, latent2, epsilon, interpolation_method=interpolation_method)

    if conditional:
        labels = torch.randint(0, generator.num_classes, (num_samples,)).to(device)

    if sim_net is None:
        sim_net = create_sample_similarity(
            "lpips-vgg16", sample_similarity_resize=resize, cuda=device == "cuda", verbose=False
        )

    with torch.inference_mode():
        distances = []
        num_batches = math.ceil(num_samples / batch_size)
        for batch_idx in range(num_batches):
            batch_latent1 = latent1[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)
            batch_latent2 = latent2[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)

            if conditional:
                batch_labels = labels[batch_idx * batch_size : (batch_idx + 1) * batch_size].to(device)
                outputs = generator(
                    torch.cat((batch_latent1, batch_latent2), dim=0), torch.cat((batch_labels, batch_labels), dim=0)
                )
            else:
                outputs = generator(torch.cat((batch_latent1, batch_latent2), dim=0))

            out1, out2 = outputs.chunk(2, dim=0)

            similarity = sim_net(out1, out2)
            dist = similarity / epsilon**2
            distances.append(dist.detach().cpu())

        distances = torch.cat(distances)

        lower = torch.quantile(distances, lower_discard, interpolation="lower") if lower_discard is not None else 0.0
        upper = torch.quantile(distances, upper_discard, interpolation="lower") if upper_discard is not None else 1.0
        distances = distances[(distances >= lower) & (distances <= upper)]

        return distances.mean(), distances.std(), distances
