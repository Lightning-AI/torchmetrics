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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from torch import Tensor

from torchmetrics import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ClasswiseWrapper.plot"]


class ClasswiseWrapper(Metric):
    """Wrapper metric for altering the output of classification metrics.

    This metric works together with classification metrics that returns multiple values (one value per class) such that
    label information can be automatically included in the output.

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
        labels: list of strings indicating the different classes.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_0': tensor(0.5000),
        'multiclassaccuracy_1': tensor(0.7500),
        'multiclassaccuracy_2': tensor(0.)}

    Example (labels as list of strings):
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(
        ...    MulticlassAccuracy(num_classes=3, average=None),
        ...    labels=["horse", "fish", "dog"]
        ... )
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.3333),
        'multiclassaccuracy_fish': tensor(0.6667),
        'multiclassaccuracy_dog': tensor(0.)}

    Example (in metric collection):
        >>> from torchmetrics import MetricCollection
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall
        >>> labels = ["horse", "fish", "dog"]
        >>> metric = MetricCollection(
        ...     {'multiclassaccuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), labels),
        ...     'multiclassrecall': ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None), labels)}
        ... )
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.),
         'multiclassaccuracy_fish': tensor(0.3333),
         'multiclassaccuracy_dog': tensor(0.4000),
         'multiclassrecall_horse': tensor(0.),
         'multiclassrecall_fish': tensor(0.3333),
         'multiclassrecall_dog': tensor(0.4000)}
    """

    def __init__(self, metric: Metric, labels: Optional[List[str]] = None) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        if labels is not None and not (isinstance(labels, list) and all(isinstance(lab, str) for lab in labels)):
            raise ValueError(f"Expected argument `labels` to either be `None` or a list of strings but got {labels}")
        self.metric = metric
        self.labels = labels
        self._update_count = 1

    def _convert(self, x: Tensor) -> Dict[str, Any]:
        name = self.metric.__class__.__name__.lower()
        if self.labels is None:
            return {f"{name}_{i}": val for i, val in enumerate(x)}
        return {f"{name}_{lab}": val for lab, val in zip(self.labels, x)}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate on batch and accumulate to global state."""
        return self._convert(self.metric(*args, **kwargs))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update state."""
        self.metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        """Compute metric."""
        return self._convert(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()

    def _wrap_update(self, update: Callable) -> Callable:
        """Overwrite to do nothing."""
        return update

    def _wrap_compute(self, compute: Callable) -> Callable:
        """Overwrite to do nothing."""
        return compute

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
            >>> from torchmetrics import ClasswiseWrapper
            >>> from torchmetrics.classification import MulticlassAccuracy
            >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
            >>> metric.update(torch.randint(3, (20,)), torch.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics import ClasswiseWrapper
            >>> from torchmetrics.classification import MulticlassAccuracy
            >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.randint(3, (20,)), torch.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
