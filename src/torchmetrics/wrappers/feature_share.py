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
from functools import lru_cache
from typing import Any, Optional, Union

from torch.nn import Module

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn

__doctest_requires__ = {("FeatureShare",): ["torch_fidelity"]}


class NetworkCache(Module):
    """Create a cached version of a network to be shared between metrics.

    Because the different metrics may invoke the same network multiple times, we can save time by caching the input-
    output pairs of the network.

    """

    def __init__(self, network: Module, max_size: int = 100) -> None:
        super().__init__()
        self.max_size = max_size
        self.network = network
        self.network.forward = lru_cache(maxsize=self.max_size)(network.forward)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the network with the given arguments."""
        return self.network(*args, **kwargs)


class FeatureShare(MetricCollection):
    """Specialized metric collection that facilitates sharing features between metrics.

    Certain metrics rely on an underlying expensive neural network for feature extraction when computing the metric.
    This wrapper allows to share the feature extraction between multiple metrics, which can save a lot of time and
    memory. This is achieved by making a shared instance of the network between the metrics and secondly by caching
    the input-output pairs of the network, such the subsequent calls to the network with the same input will be much
    faster.

    Args:
        metrics: One of the following:

            * list or tuple (sequence): if metrics are passed in as a list or tuple, will use the metrics class name
              as key for output dict. Therefore, two metrics of the same class cannot be chained this way.


            * dict: if metrics are passed in as a dict, will use each key in the dict as key for output dict.
              Use this format if you want to chain together multiple of the same metric with different parameters.
              Note that the keys in the output dict will be sorted alphabetically.

        max_cache_size: maximum number of input-output pairs to cache per metric. By default, this is none which means
            that the cache will be set to the number of metrics in the collection meaning that all features will be
            cached and shared across all metrics per batch.

    Example::
        >>> import torch
        >>> from torchmetrics.wrappers import FeatureShare
        >>> from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
        >>> # initialize the metrics
        >>> fs = FeatureShare([FrechetInceptionDistance(), KernelInceptionDistance(subset_size=10, subsets=2)])
        >>> # update metric
        >>> fs.update(torch.randint(255, (50, 3, 64, 64), dtype=torch.uint8), real=True)
        >>> fs.update(torch.randint(255, (50, 3, 64, 64), dtype=torch.uint8), real=False)
        >>> # compute metric
        >>> fs.compute()
        {'FrechetInceptionDistance': tensor(15.1700), 'KernelInceptionDistance': (tensor(-0.0012), tensor(0.0014))}

    """

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], dict[str, Metric]],
        max_cache_size: Optional[int] = None,
    ) -> None:
        # disable compute groups because the feature sharing is more custom
        super().__init__(metrics=metrics, compute_groups=False)  # type: ignore

        if max_cache_size is None:
            max_cache_size = len(self)
        if not isinstance(max_cache_size, int):
            raise TypeError(f"max_cache_size should be an integer, but got {max_cache_size}")

        try:
            first_net = next(iter(self.values()))
            if not isinstance(first_net.feature_network, str):
                raise TypeError("The `feature_network` attribute must be a string.")
            network_to_share = getattr(first_net, first_net.feature_network)
        except AttributeError as err:
            raise AttributeError(
                "Tried to extract the network to share from the first metric, but it did not have a `feature_network`"
                " attribute. Please make sure that the metric has an attribute with that name,"
                " else it cannot be shared."
            ) from err
        except TypeError as err:
            raise TypeError("The `feature_network` attribute must be a string representing the network name.") from err
        cached_net = NetworkCache(network_to_share, max_size=max_cache_size)

        # set the cached network to all metrics
        for metric_name, metric in self.items():
            if not hasattr(metric, "feature_network"):
                raise AttributeError(
                    "Tried to set the cached network to all metrics, but one of the metrics did not have a"
                    " `feature_network` attribute. Please make sure that all metrics have a attribute with that name,"
                    f" else it cannot be shared. Failed on metric {metric_name}."
                )
            if not isinstance(metric.feature_network, str):
                raise TypeError(f"Metric {metric_name}'s `feature_network` attribute must be a string.")

            # check if its the same network as the first metric
            if str(getattr(metric, metric.feature_network)) != str(network_to_share):
                rank_zero_warn(
                    f"The network to share between the metrics is not the same for all metrics."
                    f" Metric {metric_name} has a different network than the first metric."
                    " This may lead to unexpected behavior.",
                    UserWarning,
                )

            setattr(metric, metric.feature_network, cached_net)
