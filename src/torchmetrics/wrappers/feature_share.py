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
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Union

from torch.nn import Module

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric


class NetworkCache(Module):
    """Create a cached version of a network to be shared between metrics.

    Because the different metrics invoke the same network multiple times, we can save time by caching the input-output
    pairs of the network.

    """

    def __init__(self, network: Module, max_size: int = 100) -> None:
        super().__init__()
        self.max_size = max_size
        self.network = lru_cache(maxsize=self.max_size)(network)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the network with the given arguments."""
        return self.network(*args, **kwargs)


class FeatureShare(MetricCollection):
    """Specialized metric collection that can be used to share features between metrics.

    Certain metrics rely on an underlying expensive neural network for feature extraction when computing the metric.
    This wrapper allows to share the feature extraction between multiple metrics, which can save a lot of time and
    memory. This is achieved by making a shared instance of the network between the metrics and secondly by caching
    the input-output pairs of the network, such the subsequent calls to the network with the same input will be much
    faster.

    Args:
        metrics: One of the following

            * list or tuple (sequence): if metrics are passed in as a list or tuple, will use the metrics class name
              as key for output dict. Therefore, two metrics of the same class cannot be chained this way.

            * arguments: similar to passing in as a list, metrics passed in as arguments will use their metric
              class name as key for the output dict.

            * dict: if metrics are passed in as a dict, will use each key in the dict as key for output dict.
              Use this format if you want to chain together multiple of the same metric with different parameters.
              Note that the keys in the output dict will be sorted alphabetically.

        network_names: name of the network attribute to share between the metrics.
        max_cache_size: maximum number of input-output pairs to cache per metric. By default, this is none which means
            that the cache will be set to the number of metrics in the collection.

    """

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        network_names: Union[str, Sequence[str]],
        max_cache_size: Optional[int] = None,
    ) -> None:
        super().__init__(metrics=metrics)

        if isinstance(network_names, str):
            network_names = [network_names] * len(self)
        if len(network_names) != len(self):
            raise ValueError(
                "The number of network names should be equal to the number of metrics,"
                f" but got {len(network_names)} and {len(self)}')"
            )

        if max_cache_size is None:
            max_cache_size = len(self)
        if not isinstance(max_cache_size, int):
            raise TypeError(f"max_cache_size should be an integer, but got {max_cache_size}")

        # get the network of the first metric and create a cached version
        try:
            shared_net = getattr(getattr(self, next(iter(self.keys()))), network_names[0])
        except AttributeError as e:
            raise AttributeError(
                "The indicated network name of the first metric did not match any known attribute in the metric."
                "Please make sure that the network name is correct and that the metric has a attribute with that name."
                " Consider checking the `metric.named_children()` to see the available submodules."
            ) from e

        cached_net = NetworkCache(shared_net, max_size=max_cache_size)

        # set the cached network to all metrics
        for (metric_name, metric), network_name in zip(self.items(), network_names):
            attr = getattr(metric, network_name, None)
            if attr is None:
                failed_metric = metric_name
                break

            setattr(metric, network_name, cached_net)

        if attr is None:
            raise AttributeError(
                f"The indicated network name did not match any known attribute in metric {failed_metric}. Please make"
                " sure that the network name is correct and that the metric has a attribute with that name."
                " Consider checking the `metric.named_children()` to see the available submodules."
            )
