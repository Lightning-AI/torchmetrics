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
from typing import Any, Literal, Optional, Union

from torch import Tensor, nn

from torchmetrics.functional.image.lpips import _LPIPS
from torchmetrics.functional.image.perceptual_path_length import (
    GeneratorType,
    _perceptual_path_length_validate_arguments,
    _validate_generator_model,
    perceptual_path_length,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["PerceptualPathLength"]


class PerceptualPathLength(Metric):
    r"""Computes the perceptual path length (`PPL`_) of a generator model.

    The perceptual path length can be used to measure the consistency of interpolation in latent-space models. It is
    defined as

    .. math::
        PPL = \mathbb{E}\left[\frac{1}{\epsilon^2} D(G(I(z_1, z_2, t)), G(I(z_1, z_2, t+\epsilon)))\right]

    where :math:`G` is the generator, :math:`I` is the interpolation function, :math:`D` is a similarity metric,
    :math:`z_1` and :math:`z_2` are two sets of latent points, and :math:`t` is a parameter between 0 and 1. The metric
    thus works by interpolating between two sets of latent points, and measuring the similarity between the generated
    images. The expectation is approximated by sampling :math:`z_1` and :math:`z_2` from the generator, and averaging
    the calculated distanced. The similarity metric :math:`D` is by default the `LPIPS`_ metric, but can be changed by
    setting the `sim_net` argument.

    The provided generator model must have a `sample` method with signature `sample(num_samples: int) -> Tensor` where
    the returned tensor has shape `(num_samples, z_size)`. If the generator is conditional, it must also have a
    `num_classes` attribute. The `forward` method of the generator must have signature `forward(z: Tensor) -> Tensor`
    if `conditional=False`, and `forward(z: Tensor, labels: Tensor) -> Tensor` if `conditional=True`. The returned
    tensor should have shape `(num_samples, C, H, W)` and be scaled to the range [0, 255].

    .. hint::
        Using this metric with the default feature extractor requires that ``torchvision`` is installed.
        Either install as ``pip install torchmetrics[image]`` or ``pip install torchvision``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``generator`` (:class:`~torch.nn.Module`):  Generator model, with specific requirements. See above.

    As output of `forward` and `compute` the metric returns the following output

    - ``ppl_mean`` (:class:`~torch.Tensor`): float scalar tensor with mean PPL value over distances
    - ``ppl_std`` (:class:`~torch.Tensor`): float scalar tensor with std PPL value over distances
    - ``ppl_raw`` (:class:`~torch.Tensor`): float scalar tensor with raw PPL distances

    Args:
        num_samples: Number of samples to use for the PPL computation.
        conditional: Whether the generator is conditional or not (i.e. whether it takes labels as input).
        batch_size: Batch size to use for the PPL computation.
        interpolation_method: Interpolation method to use. Choose from 'lerp', 'slerp_any', 'slerp_unit'.
        epsilon: Spacing between the points on the path between latent points.
        resize: Resize images to this size before computing the similarity between generated images.
        lower_discard: Lower quantile to discard from the distances, before computing the mean and standard deviation.
        upper_discard: Upper quantile to discard from the distances, before computing the mean and standard deviation.
        sim_net: Similarity network to use. Can be a `nn.Module` or one of 'alex', 'vgg', 'squeeze', where the three
            latter options correspond to the pretrained networks from the `LPIPS`_ paper.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torch-fidelity`` is not installed.
        ValueError:
            If ``num_samples`` is not a positive integer.
        ValueError:
            If `conditional` is not a boolean.
        ValueError:
            If ``batch_size`` is not a positive integer.
        ValueError:
            If ``interpolation_method`` is not one of 'lerp', 'slerp_any', 'slerp_unit'.
        ValueError:
            If ``epsilon`` is not a positive float.
        ValueError:
            If ``resize`` is not a positive integer.
        ValueError:
            If ``lower_discard`` is not a float between 0 and 1 or None.
        ValueError:
            If ``upper_discard`` is not a float between 0 and 1 or None.

    Example::
        >>> import torch
        >>> class DummyGenerator(torch.nn.Module):
        ...    def __init__(self, z_size) -> None:
        ...       super().__init__()
        ...       self.z_size = z_size
        ...       self.model = torch.nn.Sequential(torch.nn.Linear(z_size, 3*128*128), torch.nn.Sigmoid())
        ...    def forward(self, z):
        ...       return 255 * (self.model(z).reshape(-1, 3, 128, 128) + 1)
        ...    def sample(self, num_samples):
        ...      return torch.randn(num_samples, self.z_size)
        >>> generator = DummyGenerator(2)
        >>> ppl = PerceptualPathLength(num_samples=10)
        >>> ppl(generator)
        (tensor(...), tensor(...), tensor([...]))

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    net: nn.Module
    feature_network: str = "net"

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
        sim_net: Union[nn.Module, Literal["alex", "vgg", "squeeze"]] = "vgg",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "Metric `PerceptualPathLength` requires torchvision which is not installed."
                "Install with `pip install torchvision` or `pip install torchmetrics[image]`"
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

        if isinstance(sim_net, nn.Module):
            self.net = sim_net
        elif sim_net in ["alex", "vgg", "squeeze"]:
            self.net = _LPIPS(pretrained=True, net=sim_net, resize=resize)
        else:
            raise ValueError(f"sim_net must be a nn.Module or one of 'alex', 'vgg', 'squeeze', got {sim_net}")

    def update(self, generator: GeneratorType) -> None:
        """Update the generator model."""
        _validate_generator_model(generator, self.conditional)
        self.generator = generator

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the perceptual path length."""
        return perceptual_path_length(
            generator=self.generator,
            num_samples=self.num_samples,
            conditional=self.conditional,
            interpolation_method=self.interpolation_method,
            epsilon=self.epsilon,
            resize=self.resize,
            lower_discard=self.lower_discard,
            upper_discard=self.upper_discard,
            sim_net=self.net,
            device=self.device,
        )
