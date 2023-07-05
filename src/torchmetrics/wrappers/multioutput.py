from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList

from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MultioutputWrapper.plot"]


def _get_nan_indices(*tensors: Tensor) -> Tensor:
    """Get indices of rows along dim 0 which have NaN values."""
    if len(tensors) == 0:
        raise ValueError("Must pass at least one tensor as argument")
    sentinel = tensors[0]
    nan_idxs = torch.zeros(len(sentinel), dtype=torch.bool, device=sentinel.device)
    for tensor in tensors:
        permuted_tensor = tensor.flatten(start_dim=1)
        nan_idxs |= torch.any(torch.isnan(permuted_tensor), dim=1)
    return nan_idxs


class MultioutputWrapper(Metric):
    """Wrap a base metric to enable it to support multiple outputs.

    Several torchmetrics metrics, such as :class:`torchmetrics.regression.spearman.SpearmanCorrcoef` lack support for
    multioutput mode. This class wraps such metrics to support computing one metric per output.
    Unlike specific torchmetric metrics, it doesn't support any aggregation across outputs.
    This means if you set ``num_outputs`` to 2, ``.compute()`` will return a Tensor of dimension
    ``(2, ...)`` where ``...`` represents the dimensions the metric returns when not wrapped.

    In addition to enabling multioutput support for metrics that lack it, this class also supports, albeit in a crude
    fashion, dealing with missing labels (or other data). When ``remove_nans`` is passed, the class will remove the
    intersection of NaN containing "rows" upon each update for each output. For example, suppose a user uses
    `MultioutputWrapper` to wrap :class:`torchmetrics.regression.r2.R2Score` with 2 outputs, one of which occasionally
    has missing labels for classes like ``R2Score`` is that this class supports removing ``NaN`` values
    (parameter ``remove_nans``) on a per-output basis. When ``remove_nans`` is passed the wrapper will remove all rows

    Args:
        base_metric: Metric being wrapped.
        num_outputs: Expected dimensionality of the output dimension.
            This parameter is used to determine the number of distinct metrics we need to track.
        output_dim:
            Dimension on which output is expected. Note that while this provides some flexibility, the output dimension
            must be the same for all inputs to update. This applies even for metrics such as `Accuracy` where the labels
            can have a different number of dimensions than the predictions. This can be worked around if the output
            dimension can be set to -1 for both, even if -1 corresponds to different dimensions in different inputs.
        remove_nans:
            Whether to remove the intersection of rows containing NaNs from the values passed through to each underlying
            metric. Proper operation requires all tensors passed to update to have dimension ``(N, ...)`` where N
            represents the length of the batch or dataset being passed in.
        squeeze_outputs:
            If ``True``, will squeeze the 1-item dimensions left after ``index_select`` is applied.
            This is sometimes unnecessary but harmless for metrics such as `R2Score` but useful
            for certain classification metrics that can't handle additional 1-item dimensions.

    Example:
         >>> # Mimic R2Score in `multioutput`, `raw_values` mode:
         >>> import torch
         >>> from torchmetrics.wrappers import MultioutputWrapper
         >>> from torchmetrics.regression import R2Score
         >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
         >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
         >>> r2score = MultioutputWrapper(R2Score(), 2)
         >>> r2score(preds, target)
         tensor([0.9654, 0.9082])
    """

    is_differentiable = False

    def __init__(
        self,
        base_metric: Metric,
        num_outputs: int,
        output_dim: int = -1,
        remove_nans: bool = True,
        squeeze_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.metrics = ModuleList([deepcopy(base_metric) for _ in range(num_outputs)])
        self.output_dim = output_dim
        self.remove_nans = remove_nans
        self.squeeze_outputs = squeeze_outputs

    def _get_args_kwargs_by_output(self, *args: Tensor, **kwargs: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Get args and kwargs reshaped to be output-specific and (maybe) having NaNs stripped out."""
        args_kwargs_by_output = []
        for i in range(len(self.metrics)):
            selected_args = apply_to_collection(
                args, Tensor, torch.index_select, dim=self.output_dim, index=torch.tensor(i, device=self.device)
            )
            selected_kwargs = apply_to_collection(
                kwargs, Tensor, torch.index_select, dim=self.output_dim, index=torch.tensor(i, device=self.device)
            )
            if self.remove_nans:
                args_kwargs = selected_args + tuple(selected_kwargs.values())
                nan_idxs = _get_nan_indices(*args_kwargs)
                selected_args = [arg[~nan_idxs] for arg in selected_args]
                selected_kwargs = {k: v[~nan_idxs] for k, v in selected_kwargs.items()}

            if self.squeeze_outputs:
                selected_args = [arg.squeeze(self.output_dim) for arg in selected_args]
                selected_kwargs = {k: v.squeeze(self.output_dim) for k, v in selected_kwargs.items()}
            args_kwargs_by_output.append((selected_args, selected_kwargs))
        return args_kwargs_by_output

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update each underlying metric with the corresponding output."""
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        for metric, (selected_args, selected_kwargs) in zip(self.metrics, reshaped_args_kwargs):
            metric.update(*selected_args, **selected_kwargs)

    def compute(self) -> Tensor:
        """Compute metrics."""
        return torch.stack([m.compute() for m in self.metrics], 0)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call underlying forward methods and aggregate the results if they're non-null.

        We override this method to ensure that state variables get copied over on the underlying metrics.
        """
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        results = [
            metric(*selected_args, **selected_kwargs)
            for metric, (selected_args, selected_kwargs) in zip(self.metrics, reshaped_args_kwargs)
        ]
        if results[0] is None:
            return None
        return torch.stack(results, 0)

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.metrics:
            metric.reset()
        super().reset()

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
            >>> from torchmetrics.wrappers import MultioutputWrapper
            >>> from torchmetrics.regression import R2Score
            >>> metric = MultioutputWrapper(R2Score(), 2)
            >>> metric.update(torch.randn(20, 2), torch.randn(20, 2))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.wrappers import MultioutputWrapper
            >>> from torchmetrics.regression import R2Score
            >>> metric = MultioutputWrapper(R2Score(), 2)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.randn(20, 2), torch.randn(20, 2)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
