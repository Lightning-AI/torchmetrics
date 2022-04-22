# Reference code: https://github.com/jxgu1016/Total_Variation_Loss.pytorch
import torch
from torchmetrics.metric import Metric


class TotalVariation(Metric):
    """A method to calculate total variation loss.

    .. note::

        Because this loss uses sums, the value will be large. Use a weighting
        of order e-5 to control it. Ensure to train with half-precision at least.

    :param dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
        before returning the value at the step.
    :type dist_sync_on_step: bool
    :param compute_on_step: Forward only calls ``update()`` and returns None if this is set to
        False.
    :type compute_on_step: bool
    """

    is_differentiable = True
    higher_is_better = False
    current: torch.Tensor
    total: torch.Tensor

    def __init__(self, dist_sync_on_step: bool = False, compute_on_step: bool = True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("current", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, sample: torch.Tensor) -> None:
        """Update method for TV Loss.

        :param sample: A NCHW image batch.
        :type sample: torch.Tensor

        :returns: A loss scalar.
        :rtype: torch.Tensor
        """
        _height = sample.size()[2]
        _width = sample.size()[3]
        _count_height = self.tensor_size(sample[:, :, 1:, :])
        _count_width = self.tensor_size(sample[:, :, :, 1:])
        _height_tv = torch.pow((sample[:, :, 1:, :] - sample[:, :, : _height - 1, :]), 2).sum()
        _width_tv = torch.pow((sample[:, :, :, 1:] - sample[:, :, :, : _width - 1]), 2).sum()
        self.current += 2 * (_height_tv / _count_height + _width_tv / _count_width)
        self.total += sample.numel()

    def compute(self):
        return self.current.float() / self.total

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
