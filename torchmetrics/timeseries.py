from torch import nn

from torchmetrics.metric import Metric

class TimeSeriesMetric(nn.ModuleList):
    """
    A class that keeps track of metrics as a timeseries, and implements useful methods.

    Args:
        metric: the torchmetric to keep track of at each timestep.

    """

    def __init__(self, metric: Metric):
        super().__init__()

    @property
    def n_timesteps(self) -> int:
        return len(self)
