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
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import adaptive_avg_pool2d
import torchvision
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["FrechetDistanceDinoV2.plot"]

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_dinov2 import FeatureExtractorDinoV2 as _FeatureExtractorDinoV2
    from torch_fidelity.helpers import vassert
    from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x
else:

    class _FeatureExtractorDinoV2(Module):  # type: ignore[no-redef]
        pass

    vassert = None
    interpolate_bilinear_2d_like_tensorflow1x = None

    __doctest_skip__ = ["FrechetDistanceDinoV2", "FrechetDistanceDinoV2.plot"]


class NoTrainDinoV2(_FeatureExtractorDinoV2):
    """Module that never leaves evaluation mode."""

    def __init__(
        self,
        name: str,
        features_list: list[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        if not _TORCH_FIDELITY_AVAILABLE:
            raise ModuleNotFoundError(
                "NoTrainDinoV2 module requires that `Torch-fidelity` is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
            )

        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "NoTrainDinoV2":
        """Force network to always be in evaluation mode."""
        return super().train(False)

    def _torch_fidelity_forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Forward method of dinov2 net.

        Copy of the forward method from this file:
        https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_dinov2.py
        with a single line change regarding the casting of `x` in the beginning.

        Corresponding license file (Apache License, Version 2.0):
        https://github.com/toshas/torch-fidelity/blob/master/LICENSE.md

        """
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8, "Expecting image as torch.Tensor with dtype=torch.uint8")
        vassert(x.dim() == 4 and x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")

        x = x.to(self.feature_extractor_internal_dtype)
        # N x 3 x ? x ?

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        # N x 3 x 224 x 224

        x = torchvision.transforms.functional.normalize(
            x,
            (255 * 0.485, 255 * 0.456, 255 * 0.406),
            (255 * 0.229, 255 * 0.224, 255 * 0.225),
            inplace=False,
        )
        # N x 3 x 224 x 224

        x = self.model(x)

        out = {
            "dinov2": x.to(torch.float32),
        }

        return tuple(out[a] for a in self.features_list)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of neural network with reshaping of output."""
        out = self._torch_fidelity_forward(x)
        return out[0].reshape(x.shape[0], -1)


def _compute_fd_dinov2(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fd Dinov2 Score`_.

    The Frechet Distance DinoV2 between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FrechetDistanceDinoV2(Metric):
    r"""Calculate Fréchet distance Dinov2 (FD_Dinov2_) which is used to assess the quality of generated images.

    .. math::
        FD Dinov2 = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Dino v2
    (`fd dino_v2 ref1`_) features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the
    multivariate normal distribution estimated from Dino v2 features calculated on generated (fake) images.
    The metric was originally proposed in `fd_dinov2 ref1`_.

    Using the default feature extraction (Dino v2 using the original weights from `fd_dinov2 ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 224 x 224 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    Using custom feature extractor is also possible. One can give a torch.nn.Module as `feature` argument. This
    custom feature extractor is expected to have output shape of ``(1, num_features)``. This would change the
    used feature extractor from default (Dino v2) to the given network. In case network doesn't have
    ``num_features`` attribute, a random tensor will be given to the network to infer feature dimensionality.
    Size of this tensor can be controlled by ``input_img_size`` argument and type of the tensor can be controlled
    with ``normalize`` argument (``True`` uses float32 tensors and ``False`` uses int8 tensors). In this case, update
    method expects to have the tensor given to `imgs` argument to be in the correct shape and type that is compatible
    to the custom feature extractor.

    This metric is known to be unstable in its calculatations, and we recommend for the best results using this metric
    that you calculate using `torch.float64` (default is `torch.float32`) which can be set using the `.set_dtype`
    method of the metric.

    .. hint::
        Using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or ``pip install torch-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fd_dinov2`` (:class:`~torch.Tensor`): float scalar tensor with mean FD Dinov2 value over samples

    Args:
        feature:
            Either an integer or ``nn.Module``:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can be cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        normalize:
            Argument for controlling the input image dtype normalization:

            - If default feature extractor is used, controls whether input imgs have values in range [0, 1] or not:

              - True: if input imgs have values ranged in [0, 1]. They are cast to int8/byte tensors.
              - False: if input imgs have values ranged in [0, 255]. No casting is done.

            - If custom feature extractor module is used, controls type of the input img tensors:

              - True: if input imgs are expected to be in the data type of torch.float32.
              - False: if input imgs are expected to be in the data type of torch.int8.
        input_img_size: tuple of integers. Indicates input img size to the custom feature extractor network if provided.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If torch version is lower than 1.9
        ModuleNotFoundError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``torch.nn.Module``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> from torch import rand
        >>> from torchmetrics.image.fd_dinov2 import FrechetDistanceDinoV2
        >>> fd_dinov2 = FrechetDistanceDinoV2(feature=1024)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 224, 224), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 224, 224), dtype=torch.uint8)
        >>> fd_dinov2.update(imgs_dist1, real=True)
        >>> fd_dinov2.update(imgs_dist2, real=False)
        >>> fd_dinov2.compute()
        tensor(12.6388)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    dinov2: Module
    feature_network: str = "dinov2"

    def __init__(
        self,
        feature: Union[int, Module] = 1024,
        reset_real_features: bool = True,
        normalize: bool = False,
        input_img_size: tuple[int, int, int] = (3, 224, 224),
        feature_extractor_weights_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize
        self.used_custom_model = False
        self.feature_name_dict = {
            384: 'dinov2-vit-s-14',
            768: 'dinov2-vit-b-14',
            1024: 'dinov2-vit-l-14',
            1536: 'dinov2-vit-g-14'
        }

        if isinstance(feature, int):
            num_features = feature
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "FrechetDistanceDinoV2 metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = (384, 768, 1024, 1536)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.dinov2 = NoTrainDinoV2(
                name=self.feature_name_dict[feature],
                features_list=['dinov2'],
                feature_extractor_weights_path=feature_extractor_weights_path,
            )

        elif isinstance(feature, Module):
            self.dinov2 = feature
            self.used_custom_model = True
            if hasattr(self.dinov2, "num_features"):
                if isinstance(self.dinov2.num_features, int):
                    num_features = self.dinov2.num_features
                elif isinstance(self.dinov2.num_features, Tensor):
                    num_features = int(self.dinov2.num_features.item())
                else:
                    raise TypeError("Expected `self.dinov2.num_features` to be of type int or Tensor.")
            else:
                if self.normalize:
                    dummy_image = torch.rand(1, *input_img_size, dtype=torch.float32)
                else:
                    dummy_image = torch.randint(0, 255, (1, *input_img_size), dtype=torch.uint8)
                num_features = self.dinov2(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_num_feats = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            imgs: Input img tensors to evaluate. If used custom feature extractor please
                make sure dtype and size is correct for the model.
            real: Whether given image is real or fake.

        """
        imgs = (imgs * 255).byte() if self.normalize and (not self.used_custom_model) else imgs
        features = self.dinov2(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FD Dinov2 score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FD Dinov2 score.")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fd_dinov2(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string

        """
        out = super().set_dtype(dst_type)
        if isinstance(out.dinov2, NoTrainDinoV2):
            out.dinov2._dtype = dst_type
        return out

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.fd_dinov2 import FrechetDistanceDinoV2
            >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 224, 224), dtype=torch.uint8)
            >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 224, 224), dtype=torch.uint8)
            >>> metric = FrechetDistanceDinoV2(feature=384)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.fd_dinov2 import FrechetDistanceDinoV2
            >>> imgs_dist1 = lambda: torch.randint(0, 200, (100, 3, 224, 224), dtype=torch.uint8)
            >>> imgs_dist2 = lambda: torch.randint(100, 255, (100, 3, 224, 224), dtype=torch.uint8)
            >>> metric = FrechetDistanceDinoV2(feature=1024)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)