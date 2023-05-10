from typing import Any, Callable, Dict, Optional, Sequence, Union

from torch import Tensor, nn

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MultitaskWrapper.plot"]


class MultitaskWrapper(Metric):
    """Wrapper class for computing different metrics on different tasks in the context of multitask learning.

    In multitask learning the different tasks requires different metrics to be evaluated. This wrapper allows
    for easy evaluation in such cases by supporting multiple predictions and targets through a dictionary.
    Note that only metrics where the signature of `update` follows the stardard `preds, target` is supported.

    Args:
        task_metrics:
            Dictionary associating each task to a Metric or a MetricCollection. The keys of the dictionary represent the
            names of the tasks, and the values represent the metrics to use for each task.

    Raises:
        TypeError:
            If argument `task_metrics` is not an dictionary
        TypeError:
            If not all values in the `task_metrics` dictionary is instances of `Metric` or `MetricCollection`

    Example (with a single metric per class):
         >>> import torch
         >>> from torchmetrics.wrappers import MultitaskWrapper
         >>> from torchmetrics.regression import MeanSquaredError
         >>> from torchmetrics.classification import BinaryAccuracy
         >>>
         >>> classification_target = torch.tensor([0, 1, 0])
         >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
         >>> targets = {"Classification": classification_target, "Regression": regression_target}
         >>>
         >>> classification_preds = torch.tensor([0, 0, 1])
         >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
         >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
         >>>
         >>> metrics = MultitaskWrapper({
         ...     "Classification": BinaryAccuracy(),
         ...     "Regression": MeanSquaredError()
         ... })
         >>> metrics.update(preds, targets)
         >>> metrics.compute()
         {'Classification': tensor(0.3333), 'Regression': tensor(0.8333)}

    Example (with several metrics per task):
         >>> import torch
         >>> from torchmetrics import MetricCollection
         >>> from torchmetrics.wrappers import MultitaskWrapper
         >>> from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
         >>> from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
         >>>
         >>> classification_target = torch.tensor([0, 1, 0])
         >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
         >>> targets = {"Classification": classification_target, "Regression": regression_target}
         >>>
         >>> classification_preds = torch.tensor([0, 0, 1])
         >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
         >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
         >>>
         >>> metrics = MultitaskWrapper({
         ...     "Classification": MetricCollection(BinaryAccuracy(), BinaryF1Score()),
         ...     "Regression": MetricCollection(MeanSquaredError(), MeanAbsoluteError())
         ... })
         >>> metrics.update(preds, targets)
         >>> metrics.compute()
         {'Classification': {'BinaryAccuracy': tensor(0.3333), 'BinaryF1Score': tensor(0.)},
          'Regression': {'MeanSquaredError': tensor(0.8333), 'MeanAbsoluteError': tensor(0.6667)}}
    """

    is_differentiable = False

    def __init__(
        self,
        task_metrics: Dict[str, Union[Metric, MetricCollection]],
    ) -> None:
        self._check_task_metrics_type(task_metrics)
        super().__init__()
        self.task_metrics = nn.ModuleDict(task_metrics)

    @staticmethod
    def _check_task_metrics_type(task_metrics: Dict[str, Union[Metric, MetricCollection]]) -> None:
        if not isinstance(task_metrics, dict):
            raise TypeError(f"Expected argument `task_metrics` to be a dict. Found task_metrics = {task_metrics}")

        for metric in task_metrics.values():
            if not (isinstance(metric, (Metric, MetricCollection))):
                raise TypeError(
                    "Expected each task's metric to be a Metric or a MetricCollection. "
                    f"Found a metric of type {type(metric)}"
                )

    def update(self, task_preds: Dict[str, Tensor], task_targets: Dict[str, Tensor]) -> None:
        """Update each task's metric with its corresponding pred and target.

        Args:
            task_preds: Dictionary associating each task to a Tensor of pred.
            task_targets: Dictionary associating each task to a Tensor of target.
        """
        if not self.task_metrics.keys() == task_preds.keys() == task_targets.keys():
            raise ValueError(
                "Expected arguments `task_preds` and `task_targets` to have the same keys as the wrapped `task_metrics`"
                f". Found task_preds.keys() = {task_preds.keys()}, task_targets.keys() = {task_targets.keys()} "
                f"and self.task_metrics.keys() = {self.task_metrics.keys()}"
            )

        for task_name, metric in self.task_metrics.items():
            pred = task_preds[task_name]
            target = task_targets[task_name]
            metric.update(pred, target)

    def compute(self) -> Dict[str, Any]:
        """Compute metrics for all tasks."""
        return {task_name: metric.compute() for task_name, metric in self.task_metrics.items()}

    def forward(self, task_preds: Dict[str, Tensor], task_targets: Dict[str, Tensor]) -> Dict[str, Any]:
        """Call underlying forward methods for all tasks and return the result as a dictionary."""
        # This method is overriden because we do not need the complex version defined in Metric, that relies on the
        # value of full_state_update, and that also accumulates the results. Here, all computations are handled by the
        # underlying metrics, which all have their own value of full_state_update, and which all accumulate the results
        # by themselves.
        return {
            task_name: metric(task_preds[task_name], task_targets[task_name])
            for task_name, metric in self.task_metrics.items()
        }

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.task_metrics.values():
            metric.reset()
        super().reset()

    def _wrap_update(self, update: Callable) -> Callable:
        """Overwrite to do nothing."""
        return update

    def _wrap_compute(self, compute: Callable) -> Callable:
        """Overwrite to do nothing."""
        return compute

    def plot(
        self, val: Optional[Union[Dict, Sequence[Dict]]] = None, ax: Optional[Sequence[_AX_TYPE]] = None
    ) -> Sequence[_PLOT_OUT_TYPE]:
        """TODO."""
        val = val if val is not None else self.compute()
        fig_axs = []
        for i, (task_name, task_metric) in enumerate(self.task_metrics.items()):
            if isinstance(val, Dict):
                f, a = task_metric.plot(val[task_name], ax=ax[i] if ax is not None else ax)
            elif isinstance(val, Sequence):
                f, a = task_metric.plot([v[task_name] for v in val], ax=ax[i] if ax is not None else ax)
            else:
                raise TypeError(
                    "Expected argument `val` to be None or of type Dict or Sequence[Dict]. "
                    f"Found type(val)= {type(val)}"
                )
            fig_axs.append((f, a))
        return fig_axs
