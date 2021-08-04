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

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn


class MetricCollection(nn.ModuleDict):
    """MetricCollection class can be used to chain metrics that have the same call pattern into one single class.

    Args:
        metrics: One of the following

            * list or tuple (sequence): if metrics are passed in as a list or tuple, will use the metrics class name
              as key for output dict. Therefore, two metrics of the same class cannot be chained this way.

            * arguments: similar to passing in as a list, metrics passed in as arguments will use their metric
              class name as key for the output dict.

            * dict: if metrics are passed in as a dict, will use each key in the dict as key for output dict.
              Use this format if you want to chain together multiple of the same metric with different parameters.
              Note that the keys in the output dict will be sorted alphabetically.

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
            If ``metrics`` is ``dict`` and additional_metrics are passed in.
        ValueError:
            If ``prefix`` is set and it is not a string.
        ValueError:
            If ``postfix`` is set and it is not a string.

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

    Example (input as arguments):
        >>> metrics = MetricCollection(Accuracy(), Precision(num_classes=3, average='macro'),
        ...                            Recall(num_classes=3, average='macro'))
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
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.add_metrics(metrics, *additional_metrics)

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Iteratively call forward for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        return {k: m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items()}

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Iteratively call update for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        for _, m in self.items(keep_base=True):
            m_kwargs = m._filter_kwargs(**kwargs)
            m.update(*args, **m_kwargs)

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.items()}

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        for _, m in self.items(keep_base=True):
            m.reset()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MetricCollection":
        """Make a copy of the metric collection
        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict

        """
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for _, m in self.items(keep_base=True):
            m.persistent(mode)

    def add_metrics(
        self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]], *additional_metrics: Metric
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            # set compatible with original type expectations
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            # prepare for optional additions
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                (metrics if isinstance(m, Metric) else remain).append(m)

            if remain:
                rank_zero_warn(
                    f"You have passes extra arguments {remain} which are not `Metric` so they will be ignored."
                )
        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible"
                f" with first passed dictionary {metrics} so they will be ignored."
            )

        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of `pl.metrics.Metric`"
                    )
                self[name] = metric
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, Metric):
                    raise ValueError(f"Input {metric} to `MetricCollection` is not a instance of `pl.metrics.Metric`")
                name = metric.__class__.__name__
                if name in self:
                    raise ValueError(f"Encountered two metrics both named {name}")
                self[name] = metric
        else:
            raise ValueError("Unknown input to MetricCollection.")

    def _set_name(self, base: str) -> str:
        name = base if self.prefix is None else self.prefix + base
        name = name if self.postfix is None else name + self.postfix
        return name

    def _to_renamed_ordered_dict(self) -> OrderedDict:
        od = OrderedDict()
        for k, v in self._modules.items():
            od[self._set_name(k)] = v
        return od

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        r"""Return an iterable of the ModuleDict key.
        Args:
            keep_base: Whether to add prefix/postfix on the items collection.
        """
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_ordered_dict().keys()

    def items(self, keep_base: bool = False) -> Iterable[Tuple[str, nn.Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        Args:
            keep_base: Whether to add prefix/postfix on the items collection.
        """
        if keep_base:
            return self._modules.items()
        return self._to_renamed_ordered_dict().items()

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"
