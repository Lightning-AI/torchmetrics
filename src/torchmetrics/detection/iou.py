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
from collections.abc import Sequence
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["IntersectionOverUnion", "IntersectionOverUnion.plot"]
elif not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["IntersectionOverUnion.plot"]


class IntersectionOverUnion(Metric):
    r"""Computes Intersection Over Union (IoU).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes``
          detection boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - labels: ``IntTensor`` of shape ``(num_boxes)`` containing 0-indexed detection classes for
          the boxes.

    - ``target`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground
          truth boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``iou_dict``: A dictionary containing the following key-values:

        - iou: (:class:`~torch.Tensor`)
        - iou/cl_{cl}: (:class:`~torch.Tensor`), if argument ``class metrics=True``

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
        respect_labels:
            Ignore values from boxes that do not have the same label as the ground truth box. Else will compute Iou
                between all pairs of boxes.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::

        >>> import torch
        >>> from torchmetrics.detection import IntersectionOverUnion
        >>> preds = [
        ...    {
        ...        "boxes": torch.tensor([
        ...             [296.55, 93.96, 314.97, 152.79],
        ...             [298.55, 98.96, 314.97, 151.79]]),
        ...        "labels": torch.tensor([4, 5]),
        ...    }
        ... ]
        >>> target = [
        ...    {
        ...        "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
        ...        "labels": torch.tensor([5]),
        ...    }
        ... ]
        >>> metric = IntersectionOverUnion()
        >>> metric(preds, target)
        {'iou': tensor(0.8614)}

    Example::

        The metric can also return the score per class:

        >>> import torch
        >>> from torchmetrics.detection import IntersectionOverUnion
        >>> preds = [
        ...    {
        ...        "boxes": torch.tensor([
        ...             [296.55, 93.96, 314.97, 152.79],
        ...             [298.55, 98.96, 314.97, 151.79]]),
        ...        "labels": torch.tensor([4, 5]),
        ...    }
        ... ]
        >>> target = [
        ...    {
        ...        "boxes": torch.tensor([
        ...               [300.00, 100.00, 315.00, 150.00],
        ...               [300.00, 100.00, 315.00, 150.00]
        ...        ]),
        ...        "labels": torch.tensor([4, 5]),
        ...    }
        ... ]
        >>> metric = IntersectionOverUnion(class_metrics=True)
        >>> metric(preds, target)
        {'iou': tensor(0.7756), 'iou/cl_4': tensor(0.6898), 'iou/cl_5': tensor(0.8614)}

    Raises:
        ModuleNotFoundError:
            If torchvision is not installed with version 0.8.0 or newer.

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    groundtruth_labels: List[Tensor]
    pred_labels: List[Tensor]
    iou_matrix: List[Tensor]
    _iou_type: str = "iou"
    _invalid_val: float = -1.0

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        respect_labels: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                f"Metric `{self._iou_type.upper()}` requires that `torchvision` is installed."
                " Please install with `pip install torchmetrics[detection]`."
            )

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")

        self.box_format = box_format
        self.iou_threshold = iou_threshold

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        if not isinstance(respect_labels, bool):
            raise ValueError("Expected argument `respect_labels` to be a boolean")
        self.respect_labels = respect_labels

        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("iou_matrix", default=[], dist_reduce_fx=None)

    @staticmethod
    def _iou_update_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _iou_update(*args, **kwargs)

    @staticmethod
    def _iou_compute_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _iou_compute(*args, **kwargs)

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Update state with predictions and targets."""
        _input_validator(preds, target, ignore_score=True)

        for p_i, t_i in zip(preds, target):
            det_boxes = self._get_safe_item_values(p_i["boxes"])
            gt_boxes = self._get_safe_item_values(t_i["boxes"])
            self.groundtruth_labels.append(t_i["labels"])
            self.pred_labels.append(p_i["labels"])

            iou_matrix = self._iou_update_fn(det_boxes, gt_boxes, self.iou_threshold, self._invalid_val)  # N x M
            if self.respect_labels:
                if det_boxes.numel() > 0 and gt_boxes.numel() > 0:
                    label_eq = p_i["labels"].unsqueeze(1) == t_i["labels"].unsqueeze(0)  # N x M
                else:
                    label_eq = torch.eye(iou_matrix.shape[0], dtype=bool, device=iou_matrix.device)  # type: ignore[call-overload]
                iou_matrix[~label_eq] = self._invalid_val
            self.iou_matrix.append(iou_matrix)

    def _get_safe_item_values(self, boxes: Tensor) -> Tensor:
        from torchvision.ops import box_convert

        boxes = _fix_empty_tensors(boxes)
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
        return boxes

    def _get_gt_classes(self) -> list:
        """Returns a list of unique classes found in ground truth and detection data."""
        if len(self.groundtruth_labels) > 0:
            return torch.cat(self.groundtruth_labels).unique().tolist()
        return []

    def compute(self) -> dict:
        """Computes IoU based on inputs passed in to ``update`` previously."""
        # compute global IoU score using only valid values.
        valid_matrices = [
            mat[mat != self._invalid_val] for mat in self.iou_matrix if torch.any(mat != self._invalid_val)
        ]
        score = torch.cat(valid_matrices, 0).mean() if valid_matrices else torch.tensor(0.0, device=self.device)
        results: dict[str, Tensor] = {f"{self._iou_type}": score}
        if torch.isnan(score):  # if no valid boxes are found
            results[f"{self._iou_type}"] = torch.tensor(0.0, device=score.device)
        if self.class_metrics:
            # union of ground truth and predicted labels
            all_labels = dim_zero_cat([dim_zero_cat(self.groundtruth_labels), dim_zero_cat(self.pred_labels)])
            classes = all_labels.unique().tolist() if all_labels.numel() > 0 else []
            for cl in classes:
                masked_iou = torch.zeros_like(score)
                observed = torch.zeros_like(score)

                for mat, gt_lab in zip(self.iou_matrix, self.groundtruth_labels):
                    scores = mat[:, gt_lab == cl]
                    valid_scores = scores[scores != self._invalid_val]
                    masked_iou += valid_scores.sum()
                    observed += valid_scores.numel()
                # return 0.0 if no valid scores are observed.
                if observed.item() == 0:
                    results.update({f"{self._iou_type}/cl_{cl}": torch.tensor(0.0, device=score.device)})
                else:
                    results.update({f"{self._iou_type}/cl_{cl}": masked_iou / observed})
        return results

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

            >>> import torch
            >>> from torchmetrics.detection import IntersectionOverUnion
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
            >>> metric = IntersectionOverUnion()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.detection import IntersectionOverUnion
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
            >>> metric = IntersectionOverUnion()
            >>> vals = []
            >>> for _ in range(20):
            ...     vals.append(metric(preds, target()))
            >>> fig_, ax_ = metric.plot(vals)

        """
        return self._plot(val, ax)
