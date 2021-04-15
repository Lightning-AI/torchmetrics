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

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import nn

from torchmetrics.metric import Metric


class MetricCollection(nn.ModuleDict):
    """
    MetricCollection class can be used to chain metrics that have the same
    call pattern into one single class.

    Args:
        metrics: One of the following

            * list or tuple: if metrics are passed in as a list, will use the
              metrics class name as key for output dict. Therefore, two metrics
              of the same class cannot be chained this way.

            * dict: if metrics are passed in as a dict, will use each key in the
              dict as key for output dict. Use this format if you want to chain
              together multiple of the same metric with different parameters.

        prefix: a string to append in front of the keys of the output dict

    Raises:
        ValueError:
            If one of the elements of ``metrics`` is not an instance of ``pl.metrics.Metric``.
        ValueError:
            If two elements in ``metrics`` have the same ``name``.
        ValueError:
            If ``metrics`` is not a ``list``, ``tuple`` or a ``dict``.

    Example (input as list):
        >>> import torch
        >>> from pprint import pprint
        >>> from torchmetrics import MetricCollection, Accuracy, Precision, Recall
        >>> target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
        >>> preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
        >>> metrics = MetricCollection([Accuracy(),
        ...                             Precision(num_classes=3, average='macro'),
        ...                             Recall(num_classes=3, average='macro')])
        >>> metrics(preds, target)
        {'Accuracy': tensor(0.1250), 'Precision': tensor(0.0667), 'Recall': tensor(0.1111)}

    Example (input as dict):
        >>> metrics = MetricCollection({'micro_recall': Recall(num_classes=3, average='micro'),
        ...                             'macro_recall': Recall(num_classes=3, average='macro')})
        >>> same_metric = metrics.clone()
        >>> pprint(metrics(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}
        >>> pprint(same_metric(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}
        >>> metrics.persistent()

    """

    def __init__(
        self,
        metrics: Union[List[Metric], Tuple[Metric], Dict[str, Metric]],
        prefix: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f"Value {metric} belonging to key {name}"
                        " is not an instance of `pl.metrics.Metric`"
                    )
                self[name] = metric
        elif isinstance(metrics, (tuple, list)):
            for metric in metrics:
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance"
                        " of `pl.metrics.Metric`"
                    )
                name = metric.__class__.__name__
                if name in self:
                    raise ValueError(f"Encountered two metrics both named {name}")
                self[name] = metric
        else:
            raise ValueError("Unknown input to MetricCollection.")

        self.prefix = self._check_prefix_arg(prefix)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:  # pylint: disable=E0202
        """
        Iteratively call forward for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        return {self._set_prefix(k): m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items()}

    def update(self, *args, **kwargs):  # pylint: disable=E0202
        """
        Iteratively call update for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        for _, m in self.items():
            m_kwargs = m._filter_kwargs(**kwargs)
            m.update(*args, **m_kwargs)

    def compute(self) -> Dict[str, Any]:
        return {self._set_prefix(k): m.compute() for k, m in self.items()}

    def reset(self) -> None:
        """ Iteratively call reset for each metric """
        for _, m in self.items():
            m.reset()

    def clone(self, prefix: Optional[str] = None) -> 'MetricCollection':
        """ Make a copy of the metric collection
        Args:
            prefix: a string to append in front of the metric keys
        """
        mc = deepcopy(self)
        mc.prefix = self._check_prefix_arg(prefix)
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Method for post-init to change if metric states should be saved to
        its state_dict
        """
        for _, m in self.items():
            m.persistent(mode)

    def _set_prefix(self, k: str) -> str:
        return k if self.prefix is None else self.prefix + k

    @staticmethod
    def _check_prefix_arg(prefix: str) -> Optional[str]:
        if prefix is not None:
            if isinstance(prefix, str):
                return prefix
            else:
                raise ValueError('Expected input `prefix` to be a string')
        return None
