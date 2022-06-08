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
import warnings
from typing import Any, Callable, List, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class BaseAggregator(Metric):
    """Base class for aggregation metrics.

    Args:
        fn: string specifying the reduction function
        default_value: default tensor value to use for the metric state
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encounted will give a RuntimeError
            - ``'warn'``: if any `nan` values are encounted will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impude any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float
    """

    value: Tensor
    is_differentiable = None
    higher_is_better = None
    full_state_update = False

    def __init__(
        self,
        fn: Union[Callable, str],
        default_value: Union[Tensor, List],
        nan_strategy: Union[str, float] = "error",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore")
        if nan_strategy not in allowed_nan_strategy and not isinstance(nan_strategy, float):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy}"
                f" but got {nan_strategy}."
            )

        self.nan_strategy = nan_strategy
        self.add_state("value", default=default_value, dist_reduce_fx=fn)

    def _cast_and_nan_check_input(self, x: Union[float, Tensor]) -> Tensor:
        """Converts input x to a tensor if not already and afterwards checks for nans that either give an error,
        warning or just ignored."""
        if not isinstance(x, Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        nans = torch.isnan(x)
        if nans.any():
            if self.nan_strategy == "error":
                raise RuntimeError("Encounted `nan` values in tensor")
            if self.nan_strategy == "warn":
                warnings.warn("Encounted `nan` values in tensor. Will be removed.", UserWarning)
                x = x[~nans]
            elif self.nan_strategy == "ignore":
                x = x[~nans]
            else:
                x[nans] = self.nan_strategy

        return x.float()

    def update(self, value: Union[float, Tensor]) -> None:  # type: ignore
        """Overwrite in child class."""
        pass

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.value


class MaxMetric(BaseAggregator):
    """Aggregate a stream of value into their maximum value.

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encounted will give a RuntimeError
            - ``'warn'``: if any `nan` values are encounted will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impude any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics import MaxMetric
        >>> metric = MaxMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(3.)
    """

    full_state_update = True

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__(
            "max",
            -torch.tensor(float("inf")),
            nan_strategy,
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:  # type: ignore
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
        """
        value = self._cast_and_nan_check_input(value)
        if value.numel():  # make sure tensor not empty
            self.value = torch.max(self.value, torch.max(value))


class MinMetric(BaseAggregator):
    """Aggregate a stream of value into their minimum value.

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encounted will give a RuntimeError
            - ``'warn'``: if any `nan` values are encounted will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impude any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics import MinMetric
        >>> metric = MinMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(1.)
    """

    full_state_update = True

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__(
            "min",
            torch.tensor(float("inf")),
            nan_strategy,
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:  # type: ignore
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
        """
        value = self._cast_and_nan_check_input(value)
        if value.numel():  # make sure tensor not empty
            self.value = torch.min(self.value, torch.min(value))


class SumMetric(BaseAggregator):
    """Aggregate a stream of value into their sum.

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encounted will give a RuntimeError
            - ``'warn'``: if any `nan` values are encounted will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impude any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics import SumMetric
        >>> metric = SumMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(6.)
    """

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__(
            "sum",
            torch.tensor(0.0),
            nan_strategy,
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:  # type: ignore
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
        """
        value = self._cast_and_nan_check_input(value)
        if value.numel():
            self.value += value.sum()


class CatMetric(BaseAggregator):
    """Concatenate a stream of values.

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encounted will give a RuntimeError
            - ``'warn'``: if any `nan` values are encounted will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impude any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics import CatMetric
        >>> metric = CatMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor([1., 2., 3.])
    """

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__("cat", [], nan_strategy, **kwargs)

    def update(self, value: Union[float, Tensor]) -> None:  # type: ignore
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
        """
        value = self._cast_and_nan_check_input(value)
        if value.numel():
            self.value.append(value)

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return dim_zero_cat(self.value)
        return self.value


class MeanMetric(BaseAggregator):
    """Aggregate a stream of value into their mean value.

    Args:
       nan_strategy: options:
            - ``'error'``: if any `nan` values are encounted will give a RuntimeError
            - ``'warn'``: if any `nan` values are encounted will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impude any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics import MeanMetric
        >>> metric = MeanMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(2.)
    """

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__(
            "sum",
            torch.tensor(0.0),
            nan_strategy,
            **kwargs,
        )
        self.add_state("weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor] = 1.0) -> None:  # type: ignore
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to `1.0` corresponding to simple
                harmonic average.
        """
        value = self._cast_and_nan_check_input(value)
        weight = self._cast_and_nan_check_input(weight)

        if value.numel() == 0:
            return
        # broadcast weight to value shape
        if hasattr(torch, "broadcast_to"):
            weight = torch.broadcast_to(weight, value.shape)
        else:
            if weight.shape == ():
                weight = torch.ones_like(value) * weight
            if weight.shape != value.shape:
                raise ValueError("Broadcasting not supported on PyTorch <1.8")

        self.value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.value / self.weight
