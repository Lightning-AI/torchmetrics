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
from torchmetrics.metric import Metric

# define compute groups for metric collection
_COMPUTE_GROUP_REGISTRY = []


def register_compute_group(*metrics):
    """Register a compute group of metrics.

    A compute group consist of metrics that share the underlying metric state meaning that only their
    `compute` method should differ. Compute groups are used in connection with MetricCollection to
    reduce the computational cost of metrics that share the underlying same metric state. Registered
    compute groups can found in the global variable `_COMPUTE_GROUP_REGISTRY`.

    Args:
        *metrics: An iterable of metrics
    """
    for m in metrics:
        if not issubclass(m, Metric):
            raise ValueError(
                "Expected all metrics in compute group to be subclass of `torchmetrics.Metric` but got {m}"
            )
    _COMPUTE_GROUP_REGISTRY.append(tuple(m.__name__ for m in metrics))
