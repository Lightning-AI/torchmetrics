from typing import Any, Dict, List, Optional

from torch import Tensor

from torchmetrics import Metric


class ClasswiseWrapper(Metric):
    """Wrapper class for altering the output of classification metrics that returns multiple values to include
    label information.

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
        labels: list of strings indicating the different classes.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics import Accuracy, ClasswiseWrapper
        >>> metric = ClasswiseWrapper(Accuracy(num_classes=3, average=None))
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)
        {'accuracy_0': tensor(0.5000), 'accuracy_1': tensor(0.7500), 'accuracy_2': tensor(0.)}

    Example (labels as list of strings):
        >>> import torch
        >>> from torchmetrics import Accuracy, ClasswiseWrapper
        >>> metric = ClasswiseWrapper(
        ...    Accuracy(num_classes=3, average=None),
        ...    labels=["horse", "fish", "dog"]
        ... )
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)
        {'accuracy_horse': tensor(0.3333), 'accuracy_fish': tensor(0.6667), 'accuracy_dog': tensor(0.)}

    Example (in metric collection):
        >>> import torch
        >>> from torchmetrics import Accuracy, ClasswiseWrapper, MetricCollection, Recall
        >>> labels = ["horse", "fish", "dog"]
        >>> metric = MetricCollection(
        ...     {'accuracy': ClasswiseWrapper(Accuracy(num_classes=3, average=None), labels),
        ...     'recall': ClasswiseWrapper(Recall(num_classes=3, average=None), labels)}
        ... )
        >>> preds = torch.randn(10, 3).softmax(dim=-1)
        >>> target = torch.randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'accuracy_horse': tensor(0.), 'accuracy_fish': tensor(0.3333), 'accuracy_dog': tensor(0.4000),
        'recall_horse': tensor(0.), 'recall_fish': tensor(0.3333), 'recall_dog': tensor(0.4000)}
    """

    def __init__(self, metric: Metric, labels: Optional[List[str]] = None) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        if labels is not None and not (isinstance(labels, list) and all(isinstance(lab, str) for lab in labels)):
            raise ValueError(f"Expected argument `labels` to either be `None` or a list of strings but got {labels}")
        self.metric = metric
        self.labels = labels

    def _convert(self, x: Tensor) -> Dict[str, Any]:
        name = self.metric.__class__.__name__.lower()
        if self.labels is None:
            return {f"{name}_{i}": val for i, val in enumerate(x)}
        return {f"{name}_{lab}": val for lab, val in zip(self.labels, x)}

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        return self._convert(self.metric.compute())
