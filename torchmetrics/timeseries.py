import torch
from torch import nn

from torchmetrics.metric import Metric

from typing import Tuple, Union, List


class TimeSeriesMetric(nn.ModuleList):
    """
    A class that keeps track of metrics as a timeseries, and implements useful methods.

    Args:
        metric: the torchmetric to keep track of at each timestep.
    """

    def __init__(self, metric: Metric):
        super().__init__()
        self.metric = metric

    @property
    def n_timesteps(self) -> int:
        return len(self)

    def add_timestep(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Adds a metric module to the module list"""
        module = self.metric()
        module.update(preds, targets)
        self.append(module)

    def best_metric(self, return_timestep: bool = False)\
            -> Union[float, Tuple[int, float]]:
        """Returns the highest metric out of all the timesteps.

        Args:
            return_timestep: True to return the timestep with the highest metric value.

        Returns:
            The best metric value, and optionally the timestep.
        """
        pass

    def all_metrics(self) -> List[float]:
        """Returns metrics for all timesteps. """
        return [metric.compute() for metric in self]
