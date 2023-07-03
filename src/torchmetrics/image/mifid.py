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
from typing import Any, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE, _TORCH_GREATER_EQUAL_1_10
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {
    ("MemorizationInformedFrechetInceptionDistance", "MemorizationInformedFrechetInceptionDistance.plot"): [
        "torch_fidelity"
    ]
}

if not _TORCH_GREATER_EQUAL_1_10:
    __doctest_skip__ = [
        "MemorizationInformedFrechetInceptionDistance",
        "MemorizationInformedFrechetInceptionDistance.plot",
    ]

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MemorizationInformedFrechetInceptionDistance.plot"]


def _compute_cosine_distance(features1: Tensor, features2: Tensor, cosine_distance_eps: float = 0.1) -> Tensor:
    """Compute the cosine distance between two sets of features."""
    features1_nozero = features1[torch.sum(features1, dim=1) != 0]
    features2_nozero = features2[torch.sum(features2, dim=1) != 0]

    # normalize
    norm_f1 = features1_nozero / torch.norm(features1_nozero, dim=1, keepdim=True)
    norm_f2 = features2_nozero / torch.norm(features2_nozero, dim=1, keepdim=True)

    d = 1.0 - torch.abs(torch.matmul(norm_f1, norm_f2.t()))
    mean_min_d = torch.mean(d.min(dim=1).values)
    return mean_min_d if mean_min_d < cosine_distance_eps else torch.ones_like(mean_min_d)


def _mifid_compute(
    mu1: Tensor,
    sigma1: Tensor,
    features1: Tensor,
    mu2: Tensor,
    sigma2: Tensor,
    features2: Tensor,
    cosine_distance_eps: float = 0.1,
) -> Tensor:
    """Compute MIFID score given two sets of features and their statistics."""
    fid_value = _compute_fid(mu1, sigma1, mu2, sigma2)
    distance = _compute_cosine_distance(features1, features2, cosine_distance_eps)
    # secure that very small fid values does not explode the mifid
    return fid_value / (distance + 10e-15) if fid_value > 1e-8 else torch.zeros_like(fid_value)


class MemorizationInformedFrechetInceptionDistance(Metric):
    r"""Calculate Memorization-Informed Frechet Inception Distance (MIFID_).

    MIFID is a improved variation of the Frechet Inception Distance (FID_) that penalizes memorization of the training
    set by the generator. It is calculated as

    .. math::
        MIFID = \frac{FID(F_{real}, F_{fake})}{M(F_{real}, F_{fake})}

    where :math:`FID` is the normal FID score and :math:`M` is the memorization penalty. The memorization penalty
    essentially corresponds to the average minimum cosine distance between the features of the real and fake
    distribution.

    Using the default feature extraction (Inception v3 using the original weights from `fid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3 x H x W)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0, 1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    .. note:: using this metrics requires you to have ``scipy`` install. Either install as ``pip install
        torchmetrics[image]`` or ``pip install scipy``

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or
        ``pip install torch-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``mifid`` (:class:`~torch.Tensor`): float scalar tensor with mean MIFID value over samples

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
        cosine_distance_eps: Epsilon value for the cosine distance. If the cosine distance is larger than this value
            it is set to 1 and thus ignored in the MIFID calculation.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        RuntimeError:
            If ``torch`` is version less than 1.10
        ValueError:
            If ``feature`` is set to an ``int`` and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``torch.nn.Module``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example::
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
        >>> mifid = MemorizationInformedFrechetInceptionDistance(feature=64)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> mifid.update(imgs_dist1, real=True)
        >>> mifid.update(imgs_dist2, real=False)
        >>> mifid.compute()
        tensor(3003.3691)
    """
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    inception: Module

    def __init__(
        self,
        feature: Union[int, Module] = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        cosine_distance_eps: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _TORCH_GREATER_EQUAL_1_10:
            raise RuntimeError(
                "MemorizationInformedFrechetInceptionDistance metric requires PyTorch version greater or equal to 1.10"
            )

        if isinstance(feature, int):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "MemorizationInformedFrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = [64, 192, 768, 2048]
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])

        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        if not (isinstance(cosine_distance_eps, float) and 1 >= cosine_distance_eps > 0):
            raise ValueError("Argument `cosine_distance_eps` expected to be a float greater than 0 and less than 1")
        self.cosine_distance_eps = cosine_distance_eps

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        mean_real, mean_fake = torch.mean(real_features, dim=0), torch.mean(fake_features, dim=0)
        cov_real, cov_fake = torch.cov(real_features.t()), torch.cov(fake_features.t())

        return _mifid_compute(
            mean_real,
            cov_real,
            real_features,
            mean_fake,
            cov_fake,
            fake_features,
            cosine_distance_eps=self.cosine_distance_eps,
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            # remove temporarily to avoid resetting
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()

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
            >>> from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
            >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = MemorizationInformedFrechetInceptionDistance(feature=64)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
            >>> imgs_dist1 = lambda: torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = lambda: torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = MemorizationInformedFrechetInceptionDistance(feature=64)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)


        """
        return self._plot(val, ax)
