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
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import _flatten_dict, allclose

# this is just a bypass for this module name collision with build-in one
from torchmetrics.utilities.imports import OrderedDict


class MetricCollection(ModuleDict):
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

        compute_groups:
            By default the MetricCollection will try to reduce the computations needed for the metrics in the collection
            by checking if they belong to the same **compute group**. All metrics in a compute group share the same
            metric state and are therefore only different in their compute step e.g. accuracy, precision and recall
            can all be computed from the true positives/negatives and false positives/negatives. By default,
            this argument is ``True`` which enables this feature. Set this argument to `False` for disabling
            this behaviour. Can also be set to a list of lists of metrics for setting the compute groups yourself.

    .. note::
        Metric collections can be nested at initilization (see last example) but the output of the collection will
        still be a single flatten dictionary combining the prefix and postfix arguments from the nested collection.

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
        >>> from torchmetrics import MetricCollection, Accuracy, Precision, Recall, MeanSquaredError
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

    Example (specification of compute groups):
        >>> metrics = MetricCollection(
        ...     Accuracy(),
        ...     Precision(num_classes=3, average='macro'),
        ...     MeanSquaredError(),
        ...     compute_groups=[['Accuracy', 'Precision'], ['MeanSquaredError']]
        ... )
        >>> pprint(metrics(preds, target))
        {'Accuracy': tensor(0.1250), 'MeanSquaredError': tensor(2.3750), 'Precision': tensor(0.0667)}

    Example (nested metric collections):
        >>> metrics = MetricCollection([
        ...     MetricCollection([
        ...         Accuracy(num_classes=3, average='macro'),
        ...         Precision(num_classes=3, average='macro')
        ...     ], postfix='_macro'),
        ...     MetricCollection([
        ...         Accuracy(num_classes=3, average='micro'),
        ...         Precision(num_classes=3, average='micro')
        ...     ], postfix='_micro'),
        ... ], prefix='valmetrics/')
        >>> pprint(metrics(preds, target))  # doctest: +NORMALIZE_WHITESPACE
        {'valmetrics/Accuracy_macro': tensor(0.1111),
        'valmetrics/Accuracy_micro': tensor(0.1250),
        'valmetrics/Precision_macro': tensor(0.0667),
        'valmetrics/Precision_micro': tensor(0.1250)}
    """

    _groups: Dict[int, List[str]]

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True,
    ) -> None:
        super().__init__()

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self._enable_compute_groups = compute_groups
        self._groups_checked: bool = False
        self._state_is_copy: bool = False

        self.add_metrics(metrics, *additional_metrics)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Iteratively call forward for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        res = {k: m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Iteratively call update for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        # Use compute groups if already initialized and checked
        if self._groups_checked:
            for _, cg in self._groups.items():
                # only update the first member
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
                for i in range(1, len(cg)):  # copy over the update count
                    mi = getattr(self, cg[i])
                    mi._update_count = m0._update_count
            if self._state_is_copy:
                # If we have deep copied state inbetween updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False
        else:  # the first update always do per metric to form compute groups
            for _, m in self.items(keep_base=True, copy_state=False):
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def _merge_compute_groups(self) -> None:
        """Iterates over the collection of metrics, checking if the state of each metric matches another.

        If so, their compute groups will be merged into one
        """
        n_groups = len(self._groups)
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                    if cg_idx1 == cg_idx2:
                        continue

                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])

                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break

                # Start over if we merged groups
                if len(self._groups) != n_groups:
                    break

            # Stop when we iterate over everything and do not merge any groups
            if len(self._groups) == n_groups:
                break
            else:
                n_groups = len(self._groups)

        # Re-index groups
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        # empty state
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False

        if metric1._defaults.keys() != metric2._defaults.keys():
            return False

        for key in metric1._defaults.keys():
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)

            if type(state1) != type(state2):
                return False

            if isinstance(state1, Tensor) and isinstance(state2, Tensor):
                return state1.shape == state2.shape and allclose(state1, state2)

            if isinstance(state1, list) and isinstance(state2, list):
                return all(s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2))

        return True

    def _compute_groups_create_state_ref(self, copy: bool = False) -> None:
        """Create reference between metrics in the same compute group.

        Args:
            copy: If `True` the metric state will between members will be copied instead
                of just passed by reference
        """
        if not self._state_is_copy:
            for _, cg in self._groups.items():
                m0 = getattr(self, cg[0])
                for i in range(1, len(cg)):
                    mi = getattr(self, cg[i])
                    for state in m0._defaults:
                        m0_state = getattr(m0, state)
                        # Determine if we just should set a reference or a full copy
                        setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
        self._state_is_copy = copy

    def compute(self) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        res = {k: m.compute() for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        for _, m in self.items(keep_base=True, copy_state=False):
            m.reset()
        if self._enable_compute_groups and self._groups_checked:
            # reset state reference
            self._compute_groups_create_state_ref()

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
        for _, m in self.items(keep_base=True, copy_state=False):
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
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        self[f"{name}_{k}"] = v
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(f"Encountered two metrics both named {name}")
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        self[k] = v
        else:
            raise ValueError("Unknown input to MetricCollection.")

        self._groups_checked = False
        if self._enable_compute_groups:
            self._init_compute_groups()
        else:
            self._groups = {}

    def _init_compute_groups(self) -> None:
        """Initialize compute groups.

        If user provided a list, we check that all metrics in the list are also in the collection. If set to `True` we
        simply initialize each metric in the collection as its own group
        """
        if isinstance(self._enable_compute_groups, list):
            self._groups = {i: k for i, k in enumerate(self._enable_compute_groups)}
            for v in self._groups.values():
                for metric in v:
                    if metric not in self:
                        raise ValueError(
                            f"Input {metric} in `compute_groups` argument does not match a metric in the collection."
                            f" Please make sure that {self._enable_compute_groups} matches {self.keys(keep_base=True)}"
                        )
            self._groups_checked = True
        else:
            # Initialize all metrics as their own compute group
            self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}

    @property
    def compute_groups(self) -> Dict[int, List[str]]:
        """Return a dict with the current compute groups in the collection."""
        return self._groups

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
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

    def items(self, keep_base: bool = False, copy_state: bool = True) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.

        Args:
            keep_base: Whether to add prefix/postfix on the collection.
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        if keep_base:
            return self._modules.items()
        return self._to_renamed_ordered_dict().items()

    def values(self, copy_state: bool = True) -> Iterable[Module]:
        """Return an iterable of the ModuleDict values.

        Args:
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules.values()

    def __getitem__(self, key: str, copy_state: bool = True) -> Module:
        """Retrieve a single metric from the collection.

        Args:
            key: name of metric to retrieve
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules[key]

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
