from copy import deepcopy
from typing import Any, List, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleList

from torchmetrics import Metric
from torchmetrics.utilities import apply_to_collection


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
         >>> from torchmetrics import MultioutputWrapper, R2Score
         >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
         >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
         >>> r2score = MultioutputWrapper(R2Score(), 2)
         >>> r2score(preds, target)
         [tensor(0.9654), tensor(0.9082)]
         >>> # Classification metric where prediction and label tensors have different shapes.
         >>> from torchmetrics import BinnedAveragePrecision
         >>> target = torch.tensor([[1, 2], [2, 0], [1, 2]])
         >>> preds = torch.tensor([
         ...     [[.1, .8], [.8, .05], [.1, .15]],
         ...     [[.1, .1], [.2, .3], [.7, .6]],
         ...     [[.002, .4], [.95, .45], [.048, .15]]
         ... ])
         >>> binned_avg_precision = MultioutputWrapper(BinnedAveragePrecision(3, thresholds=5), 2)
         >>> binned_avg_precision(preds, target)
         [[tensor(-0.), tensor(1.0000), tensor(1.0000)], [tensor(0.3333), tensor(-0.), tensor(0.6667)]]
    """

    is_differentiable = False

    def __init__(
        self,
        base_metric: Metric,
        num_outputs: int,
        output_dim: int = -1,
        remove_nans: bool = True,
        squeeze_outputs: bool = True,
    ):
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
            args_kwargs_by_output.append((selected_args, selected_kwargs))
        return args_kwargs_by_output

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update each underlying metric with the corresponding output."""
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        for metric, (selected_args, selected_kwargs) in zip(self.metrics, reshaped_args_kwargs):
            metric.update(*selected_args, **selected_kwargs)

    def compute(self) -> List[Tensor]:
        """Compute metrics."""
        return [m.compute() for m in self.metrics]

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call underlying forward methods and aggregate the results if they're non-null.

        We override this method to ensure that state variables get copied over on the underlying metrics.
        """
        results = []
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        for metric, (selected_args, selected_kwargs) in zip(self.metrics, reshaped_args_kwargs):
            results.append(metric(*selected_args, **selected_kwargs))
        if results[0] is None:
            return None
        return results

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.metrics:
            metric.reset()
