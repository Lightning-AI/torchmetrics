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

        postfix: a string to append after the keys of the output dict

    Raises:
        ValueError:
            If one of the elements of ``metrics`` is not an instance of ``pl.metrics.Metric``.
        ValueError:
            If two elements in ``metrics`` have the same ``name``.
        ValueError:
            If ``metrics`` is not a ``list``, ``tuple`` or a ``dict``.
        ValueError:
            If ``prefix`` is set and it is not a string
        ValueError:
            If ``postfix`` is set and it is not a string

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
        postfix: Optional[str] = None
    ):
        super().__init__()
        if isinstance(metrics, dict):
            # Check all values are metrics
            for name, metric in metrics.items():
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

        self.prefix = self._check_arg(prefix, 'prefix')
        self.postfix = self._check_arg(postfix, 'postfix')

    def forward(self, *args, **kwargs) -> Dict[str, Any]:  # pylint: disable=E0202
        """
        Iteratively call forward for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        return {self._set_name(k): m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items()}

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
        return {self._set_name(k): m.compute() for k, m in self.items()}

    def reset(self) -> None:
        """ Iteratively call reset for each metric """
        for _, m in self.items():
            m.reset()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> 'MetricCollection':
        """ Make a copy of the metric collection
        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dic

        """
        mc = deepcopy(self)
        if prefix is not None:
            mc.prefix = self._check_arg(prefix, 'prefix')
        if postfix is not None:
            mc.postfix = self._check_arg(postfix, 'postfix')
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Method for post-init to change if metric states should be saved to
        its state_dict
        """
        for _, m in self.items():
            m.persistent(mode)

    def _set_name(self, k: str) -> str:
        out = k if self.prefix is None else self.prefix + k
        out = out if self.postfix is None else out + self.postfix
        return out

    @staticmethod
    def _check_arg(arg: str, name: str) -> Optional[str]:
        if arg is not None:
            if isinstance(arg, str):
                return arg
            else:
                raise ValueError(f'Expected input {name} to be a string')
        return None
