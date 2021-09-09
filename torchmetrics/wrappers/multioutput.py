from copy import deepcopy
from typing import Any, Callable, Optional

import torch
from torch import nn

from torchmetrics import Metric
from torchmetrics.utilities import apply_to_collection


def _get_nan_indices(*tensors: torch.Tensor, aligned_dim: int = 0) -> torch.Tensor:
    """Get indices of rows along `aligned_dim` which have NaN values."""
    if len(tensors) == 0:
        raise ValueError("Must pass at least one tensor as argument")
    sentinel = tensors[0]
    nan_idxs = torch.zeros(len(sentinel), dtype=torch.bool)
    for tensor in tensors:
        permuted_tensor = tensor.movedim(aligned_dim, 0).flatten(start_dim=1)
        nan_idxs |= torch.any(permuted_tensor.isnan(), dim=1)
    return nan_idxs


class MultioutputWrapper(Metric):
    """Wrap a base metric to enable it to support multiple outputs.

    Several torchmetrics metrics, such as :class:`torchmetrics.regression.spearman.SpearmanCorrcoef` lack support for
    multioutput mode. This class wraps such metrics to support computing one metric per output.
    Unlike specific torchmetric metrics, it doesn't support any
    aggregation across outputs. This means if you set `num_outputs` to 2, `compute()` will return a Tensor of dimension
    (2, ...) where ... represents the dimensions the metric returns when not wrapped.

    In addition to enabling multioutput support for metrics that lack it, this class also supports, albeit in a crude
    fashion, dealing with missing labels (or other data). When `remove_nans` is passed, the class will remove the
    intersection of NaN containing "rows" upon each update for each output. For example, suppose a user uses
    `MultioutputWrapper` to wrap :class:`torchmetrics.regression.r2.R2Score` with 2 outputs, one of which occasionally
    has missing labels for classes like `R2Score` is that this class supports removing NaN values
    (parameter `remove_nans`) on a per-output basis. When `remove_nans` is passed the wrapper will remove all rows

    Args:
        base_metric: Metric,
            Metric being wrapped.
        num_outputs: int = 1,
            Expected dimensionality of the output dimension. This parameter is
            used to determine the number of distinct metrics we need to track.
        output_dim: int = -1,
            Dimension on which output is expected. Note that while this provides some flexibility, the output dimension
            must be the same for all inputs to update. This applies even for metrics such as `Accuracy` where the labels
            can have a different number of dimensions than the predictions. This can be worked around if the output
            dimension can be set to -1 for both, even if -1 corresponds to different dimensions in different inputs.
        remove_nans: bool = True,
            Whether to remove the intersection of rows containing NaNs from the values passed through to each underlying
            metric. Proper operation requires all tensors passed to update to have dimension `(N, ...)` where N
            represents the length of the batch or dataset being passed in.
        compute_on_step: bool = True,
            Whether to recompute the metric value on each update step.
        dist_sync_on_step: bool = False,
            Required for distributed training support. See torchmetrics docs for additional details.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn: Callable = None,
            Required for distributed training support. See torchmetrics docs for additional details.
    """

    def __init__(
        self,
        base_metric: Metric,
        num_outputs: int = 1,
        output_dim: int = -1,
        remove_nans: bool = True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.metrics = nn.ModuleList([deepcopy(base_metric) for _ in range(num_outputs)])
        self.output_dim = output_dim
        self.remove_nans = remove_nans

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update each underlying metric with the corresponding output."""
        for i, metric in enumerate(self.metrics):
            selected_args = apply_to_collection(
                args, torch.Tensor, torch.index_select, dim=self.output_dim, index=torch.tensor(i)
            )
            selected_kwargs = apply_to_collection(
                kwargs, torch.Tensor, torch.index_select, dim=self.output_dim, index=torch.tensor(i)
            )
            if self.remove_nans:
                args_kwargs = selected_args + tuple(selected_kwargs.values())
                nan_idxs = _get_nan_indices(*args_kwargs)
                selected_args = [arg[~nan_idxs] for arg in selected_args]
                selected_kwargs = {k: v[~nan_idxs] for k, v in selected_kwargs.items()}
            metric.update(*selected_args, **selected_kwargs)

    def compute(self) -> torch.Tensor:
        """Compute metrics."""
        return torch.stack([m.compute() for m in self.metrics], dim=0)

    @property
    def is_differentiable(self) -> bool:
        return False

    def reset(self):
        """Reset all underlying metrics."""
        for metric in self.metrics:
            metric.reset()
