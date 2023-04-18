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
from typing import Any, Callable, Optional, Sequence, Union

from torch import Tensor

from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.functional.detection.ciou import _ciou_compute, _ciou_update
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _TORCHVISION_GREATER_EQUAL_0_13:
    __doctest_skip__ = ["CompleteIntersectionOverUnion", "CompleteIntersectionOverUnion.plot"]
elif not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CompleteIntersectionOverUnion.plot"]


class CompleteIntersectionOverUnion(IntersectionOverUnion):
    r"""Computes Complete Intersection Over Union (CIoU) <https://arxiv.org/abs/2005.03572>`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - boxes: (:class:`~torch.FloatTensor`) of shape ``(num_boxes, 4)`` containing ``num_boxes`` detection
          boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - scores: :class:`~torch.FloatTensor` of shape ``(num_boxes)`` containing detection scores for the boxes.
        - labels: :class:`~torch.IntTensor` of shape ``(num_boxes)`` containing 0-indexed detection classes for
          the boxes.

    - ``target`` (:class:`~List`) A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - boxes: :class:`~torch.FloatTensor` of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground truth
          boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - labels: :class:`~torch.IntTensor` of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``ciou_dict``: A dictionary containing the following key-values:

        - ciou: (:class:`~torch.Tensor`)
        - ciou/cl_{cl}: (:class:`~torch.Tensor`), if argument ``class metrics=True``

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.detection import CompleteIntersectionOverUnion
        >>> preds = [
        ...    {
        ...        "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
        ...        "scores": torch.tensor([0.236, 0.56]),
        ...        "labels": torch.tensor([4, 5]),
        ...    }
        ... ]
        >>> target = [
        ...    {
        ...        "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
        ...        "labels": torch.tensor([5]),
        ...    }
        ... ]
        >>> metric = CompleteIntersectionOverUnion()
        >>> metric(preds, target)
        {'ciou': tensor(-0.5694)}

    Raises:
        ModuleNotFoundError:
            If torchvision is not installed with version 0.13.0 or newer.

    """
    _iou_type: str = "ciou"
    _invalid_val: float = -2.0  # unsure, min val could be just -1.5 as well

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_GREATER_EQUAL_0_13:
            raise ModuleNotFoundError(
                f"Metric `{self._iou_type.upper()}` requires that `torchvision` version 0.13.0 or newer is installed."
                " Please install with `pip install torchvision>=0.13` or `pip install torchmetrics[detection]`."
            )
        super().__init__(box_format, iou_threshold, class_metrics, **kwargs)

    @staticmethod
    def _iou_update_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _ciou_update(*args, **kwargs)

    @staticmethod
    def _iou_compute_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _ciou_compute(*args, **kwargs)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting single value
            >>> import torch
            >>> from torchmetrics.detection import CompleteIntersectionOverUnion
            >>> preds = [
            ...    {
            ...        "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
            ...        "scores": torch.tensor([0.236, 0.56]),
            ...        "labels": torch.tensor([4, 5]),
            ...    }
            ... ]
            >>> target = [
            ...    {
            ...        "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
            ...        "labels": torch.tensor([5]),
            ...    }
            ... ]
            >>> metric = CompleteIntersectionOverUnion()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.detection import CompleteIntersectionOverUnion
            >>> preds = [
            ...    {
            ...        "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
            ...        "scores": torch.tensor([0.236, 0.56]),
            ...        "labels": torch.tensor([4, 5]),
            ...    }
            ... ]
            >>> target = lambda : [
            ...    {
            ...        "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]) + torch.randint(-10, 10, (1, 4)),
            ...        "labels": torch.tensor([5]),
            ...    }
            ... ]
            >>> metric = CompleteIntersectionOverUnion()
            >>> vals = []
            >>> for _ in range(20):
            ...     vals.append(metric(preds, target()))
            >>> fig_, ax_ = metric.plot(vals)
        """
        return self._plot(val, ax)
