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
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["KernelInceptionDistance.plot"]

__doctest_requires__ = {("KernelInceptionDistance", "KernelInceptionDistance.plot"): ["torch_fidelity"]}


def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Adapted from `KID Score`_."""
    m = k_xx.shape[0]

    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)

    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m**2)
    return value


def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    """Adapted from `KID Score`_."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree


def poly_mmd(
    f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
) -> Tensor:
    """Adapted from `KID Score`_."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)


class KernelInceptionDistance(Metric):
    r"""Calculate Kernel Inception Distance (KID) which is used to access the quality of generated images.

    .. math::
        KID = MMD(f_{real}, f_{fake})^2

    where :math:`MMD` is the maximum mean discrepancy and :math:`I_{real}, I_{fake}` are extracted features
    from real and fake images, see `kid ref1`_ for more details. In particular, calculating the MMD requires the
    evaluation of a polynomial kernel function :math:`k`

    .. math::
        k(x,y) = (\gamma * x^T y + coef)^{degree}

    which controls the distance between two features. In practise the MMD is calculated over a number of
    subsets to be able to both get the mean and standard deviation of KID.

    Using the default feature extraction (Inception v3 using the original weights from `kid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3 x H x W)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0, 1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or
        ``pip install torch-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor of shape ``(N,C,H,W)``
    - ``real`` (`bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``kid_mean`` (:class:`~torch.Tensor`): float scalar tensor with mean value over subsets
    - ``kid_std`` (:class:`~torch.Tensor`): float scalar tensor with mean value over subsets

    Args:
        feature: Either an str, integer or ``nn.Module``:

            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        subsets: Number of subsets to calculate the mean and standard deviation scores over
        subset_size: Number of randomly picked samples in each subset
        degree: Degree of the polynomial kernel function
        gamma: Scale-length of polynomial kernel. If set to ``None`` will be automatically set to the feature size
        coef: Bias term in the polynomial kernel.
        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in ``(64, 192, 768, 2048)``
        ValueError:
            If ``subsets`` is not an integer larger than 0
        ValueError:
            If ``subset_size`` is not an integer larger than 0
        ValueError:
            If ``degree`` is not an integer larger than 0
        ValueError:
            If ``gamma`` is niether ``None`` or a float larger than 0
        ValueError:
            If ``coef`` is not an float larger than 0
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.kid import KernelInceptionDistance
        >>> kid = KernelInceptionDistance(subset_size=50)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> kid.update(imgs_dist1, real=True)
        >>> kid.update(imgs_dist2, real=False)
        >>> kid_mean, kid_std = kid.compute()
        >>> print((kid_mean, kid_std))
        (tensor(0.0337), tensor(0.0023))
    """
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        feature: Union[str, int, Module] = 2048,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef: float = 1.0,
        reset_real_features: bool = True,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `Kernel Inception Distance` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "Kernel Inception Distance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception: Module = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not (isinstance(subsets, int) and subsets > 0):
            raise ValueError("Argument `subsets` expected to be integer larger than 0")
        self.subsets = subsets

        if not (isinstance(subset_size, int) and subset_size > 0):
            raise ValueError("Argument `subset_size` expected to be integer larger than 0")
        self.subset_size = subset_size

        if not (isinstance(degree, int) and degree > 0):
            raise ValueError("Argument `degree` expected to be integer larger than 0")
        self.degree = degree

        if gamma is not None and not (isinstance(gamma, float) and gamma > 0):
            raise ValueError("Argument `gamma` expected to be `None` or float larger than 0")
        self.gamma = gamma

        if not (isinstance(coef, float) and coef > 0):
            raise ValueError("Argument `coef` expected to be float larger than 0")
        self.coef = coef

        if not isinstance(reset_real_features, bool):
            raise ValueError("Arugment `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Calculate KID score based on accumulated extracted features from the two distributions.

        Implementation inspired by `Fid Score`_
        """
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        n_samples_real = real_features.shape[0]
        if n_samples_real < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")
        n_samples_fake = fake_features.shape[0]
        if n_samples_fake < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")

        kid_scores_ = []
        for _ in range(self.subsets):
            perm = torch.randperm(n_samples_real)
            f_real = real_features[perm[: self.subset_size]]
            perm = torch.randperm(n_samples_fake)
            f_fake = fake_features[perm[: self.subset_size]]

            o = poly_mmd(f_real, f_fake, self.degree, self.gamma, self.coef)
            kid_scores_.append(o)
        kid_scores = torch.stack(kid_scores_)
        return kid_scores.mean(), kid_scores.std(unbiased=False)

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
            >>> from torchmetrics.image.kid import KernelInceptionDistance
            >>> imgs_dist1 = torch.randint(0, 200, (30, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = torch.randint(100, 255, (30, 3, 299, 299), dtype=torch.uint8)
            >>> metric = KernelInceptionDistance(subsets=3, subset_size=20)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.kid import KernelInceptionDistance
            >>> imgs_dist1 = lambda: torch.randint(0, 200, (30, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = lambda: torch.randint(100, 255, (30, 3, 299, 299), dtype=torch.uint8)
            >>> metric = KernelInceptionDistance(subsets=3, subset_size=20)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute()[0])
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)
        """
        val = val or self.compute()[0]  # by default we select the mean to plot
        return self._plot(val, ax)
