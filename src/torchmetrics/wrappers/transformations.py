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
from typing import Any, Callable, Optional, Union

import torch

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric


class MetricInputTransformer(WrapperMetric):
    """Abstract base class for metric input transformations.

    Input transformations are characterized by them applying a transformation to the input data of a metric, and then
    forwarding all calls to the wrapped metric with modifications applied.

    """

    def __init__(self, wrapped_metric: Union[Metric, MetricCollection], **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        if not isinstance(wrapped_metric, (Metric, MetricCollection)):
            raise TypeError(
                f"Expected wrapped metric to be an instance of `torchmetrics.Metric` or "
                f"`torchmetrics.MetricsCollection`but received {wrapped_metric}"
            )
        self.wrapped_metric = wrapped_metric

    def transform_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Define transform operations on the prediction data.

        Overridden by subclasses. Identity by default.

        """
        return pred

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """Define transform operations on the target data.

        Overridden by subclasses. Identity by default.

        """
        return target

    def _wrap_transform(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Wrap transformation functions to dispatch args to their individual transform functions."""
        if len(args) == 1:
            return (self.transform_pred(args[0]),)
        if len(args) == 2:
            return self.transform_pred(args[0]), self.transform_target(args[1])
        return self.transform_pred(args[0]), self.transform_target(args[1]), *args[2:]

    def update(self, *args: torch.Tensor, **kwargs: dict[str, Any]) -> None:
        """Wrap the update call of the underlying metric."""
        args = self._wrap_transform(*args)
        self.wrapped_metric.update(*args, **kwargs)

    def compute(self) -> Any:
        """Wrap the compute call of the underlying metric."""
        return self.wrapped_metric.compute()

    def forward(self, *args: torch.Tensor, **kwargs: dict[str, Any]) -> Any:
        """Wrap the forward call of the underlying metric."""
        args = self._wrap_transform(*args)
        return self.wrapped_metric.forward(*args, **kwargs)


class LambdaInputTransformer(MetricInputTransformer):
    """Wrapper class for transforming a metrics' inputs given a user-defined lambda function.

    Args:
        wrapped_metric:
            The underlying `Metric` or `MetricCollection`.
        transform_pred:
            The function to apply to the predictions before computing the metric.
        transform_target:
            The function to apply to the target before computing the metric.

    Raises:
        TypeError:
            If `transform_pred` is not a Callable.
        TypeError:
            If `transform_target` is not a Callable.

    Example:
        >>> import torch
        >>> from torchmetrics.classification import BinaryAccuracy
        >>> from torchmetrics.wrappers import LambdaInputTransformer
        >>>
        >>> preds = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4])
        >>> targets = torch.tensor([1,0,0,0,0,1,1,0,0,0])
        >>>
        >>> metric = LambdaInputTransformer(BinaryAccuracy(), lambda preds: 1 - preds)
        >>> metric.update(preds, targets)
        >>> metric.compute()
        tensor(0.6000)

    """

    def __init__(
        self,
        wrapped_metric: Metric,
        transform_pred: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_target: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(wrapped_metric, **kwargs)
        if transform_pred is not None:
            if not callable(transform_pred):
                raise TypeError(f"Expected `transform_pred` to be of type `Callable` but received `{transform_pred}`")
            self.transform_pred = transform_pred  # type: ignore[assignment,method-assign]

        if transform_target is not None:
            if not callable(transform_target):
                raise TypeError(
                    f"Expected `transform_target` to be of type `Callable` but received `{transform_target}`"
                )
            self.transform_target = transform_target  # type: ignore[assignment,method-assign]


class BinaryTargetTransformer(MetricInputTransformer):
    """Wrapper class for computing a metric on binarized targets.

    Useful when the given ground-truth targets are continuous, but the metric requires binary targets.

    Args:
        wrapped_metric:
            The underlying `Metric` or `MetricCollection`.
        threshold:
            The binarization threshold for the targets. Targets values `t` are cast to binary with `t > threshold`.

    Raises:
        TypeError:
            If `threshold` is not an `int` or `float`.

    Example:
        >>> import torch
        >>> from torchmetrics.retrieval import RetrievalMRR
        >>> from torchmetrics.wrappers import BinaryTargetTransformer
        >>>
        >>> preds = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4])
        >>> targets = torch.tensor([1,0,0,0,0,2,1,0,0,0])
        >>> topics = torch.tensor([0,0,0,0,0,1,1,1,1,1])
        >>>
        >>> metric = BinaryTargetTransformer(RetrievalMRR())
        >>> metric.update(preds, targets, indexes=topics)
        >>> metric.compute()
        tensor(0.7500)

    """

    def __init__(self, wrapped_metric: Union[Metric, MetricCollection], threshold: float = 0, **kwargs: Any) -> None:
        super().__init__(wrapped_metric, **kwargs)
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"Expected `threshold` to be of type `int` or `float` but received `{threshold}`")
        self.threshold = threshold

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """Cast the target tensor to binary values according to the threshold.

        Output assumes same type as input.

        """
        return target.gt(self.threshold).to(target.dtype)
