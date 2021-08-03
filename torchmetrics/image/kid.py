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
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Adapted from https://github.com/toshas/torch- fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py."""
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
    value -= 2 * k_xy_sum / (m ** 2)
    return value


def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    """Adapted from https://github.com/toshas/torch- fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    kernel = (f1 @ f2.T * gamma + coef) ** degree
    return kernel


def poly_mmd(
    f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
) -> Tensor:
    """Adapted from https://github.com/toshas/torch- fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)


class KID(Metric):
    r"""
    Calculates Kernel Inception Distance (KID) which is used to access the quality of generated images. Given by

    .. math::
        KID = MMD(f_{real}, f_{fake})^2

    where :math:`MMD` is the maximum mean discrepancy and :math:`I_{real}, I_{fake}` are extracted features
    from real and fake images, see [1] for more details. In particular, calculating the MMD requires the
    evaluation of a polynomial kernel function :math:`k`

    .. math::
        k(x,y) = (\gamma * x^T y + coef)^{degree}

    which controls the distance between two features. In practise the MMD is calculated over a number of
    subsets to be able to both get the mean and standard deviation of KID.

    Using the default feature extraction (Inception v3 using the original weights from [2]), the input is
    expected to be mini-batches of 3-channel RGB images of shape (3 x H x W) with dtype uint8. All images
    will be resized to 299 x 299 which is the size of the original training data.

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or
        ``pip install torch-fidelity``

    .. note:: the ``forward`` method can be used but ``compute_on_step`` is disabled by default (oppesit of
        all other metrics) as this metric does not really make sense to calculate on a single batch. This
        means that by default ``forward`` will just call ``update`` underneat.

    Args:
        feature:
            Either an str, integer or ``nn.Module``:

            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``[N,d]`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        subsets:
            Number of subsets to calculate the mean and standard deviation scores over
        subset_size:
            Number of randomly picked samples in each subset
        degree:
            Degree of the polynomial kernel function
        gamma:
            Scale-length of polynomial kernel. If set to ``None`` will be automatically set to the feature size
        coef:
            Bias term in the polynomial kernel.
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    References:
        [1] Demystifying MMD GANs
        Mikołaj Bińkowski, Danica J. Sutherland, Michael Arbel, Arthur Gretton
        https://arxiv.org/abs/1801.01401

        [2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
        Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
        https://arxiv.org/abs/1706.08500

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
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

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics import KID
        >>> kid = KID(subset_size=50)  # doctest: +SKIP
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)  # doctest: +SKIP
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)  # doctest: +SKIP
        >>> kid.update(imgs_dist1, real=True)  # doctest: +SKIP
        >>> kid.update(imgs_dist2, real=False)  # doctest: +SKIP
        >>> kid_mean, kid_std = kid.compute()  # doctest: +SKIP
        >>> print((kid_mean, kid_std))  # doctest: +SKIP
        (tensor(0.0338), tensor(0.0025))

    """
    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        feature: Union[str, int, torch.nn.Module] = 2048,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,  # type: ignore
        coef: float = 1.0,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        rank_zero_warn(
            "Metric `KID` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise RuntimeError(
                    "KID metric requires that Torch-fidelity is installed."
                    " Either install as `pip install torchmetrics[image]`"
                    " or `pip install torch-fidelity`"
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}," f" but got {feature}."
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

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        features = self.inception(imgs)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Calculate KID score based on accumulated extracted features from the two distributions. Returns a tuple
        of mean and standard deviation of KID scores calculated on subsets of extracted features.

        Implementation inspired by https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py
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
