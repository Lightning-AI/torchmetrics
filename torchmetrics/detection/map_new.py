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
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

if _TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_area, box_convert, box_iou, generalized_box_iou
else:
    box_convert = None
    box_iou = None
    generalized_box_iou = None
    box_area = None

log = logging.getLogger(__name__)


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""

    map: Tensor
    map_50: Tensor
    map_75: Tensor
    map_small: Tensor
    map_medium: Tensor
    map_large: Tensor

    def __getitem__(self, key: str) -> Union[Tensor, List[Tensor]]:
        return getattr(self, key)


@dataclass
class MARMetricResults:
    """Dataclass to wrap the final mAR results."""

    mar_1: Tensor
    mar_10: Tensor
    mar_100: Tensor
    mar_small: Tensor
    mar_medium: Tensor
    mar_large: Tensor

    def __getitem__(self, key: str) -> Union[Tensor, List[Tensor]]:
        return getattr(self, key)


@dataclass
class COCOMetricResults(MAPMetricResults, MARMetricResults):
    """Dataclass to wrap the final COCO metric results including various mAP/mAR values."""

    map_per_class: Tensor
    mar_100_per_class: Tensor
    pass


def _input_validator(preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
    """Ensure the correct input format of `preds` and `targets`"""

    if not isinstance(preds, Sequence):
        raise ValueError("Expected argument `preds` to be of type List")
    if not isinstance(targets, Sequence):
        raise ValueError("Expected argument `target` to be of type List")
    if len(preds) != len(targets):
        raise ValueError("Expected argument `preds` and `target` to have the same length")

    for k in ["boxes", "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in ["boxes", "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(pred["boxes"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all boxes in `preds` to be of type torch.Tensor")
    if any(type(pred["scores"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type torch.Tensor")
    if any(type(pred["labels"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type torch.Tensor")
    if any(type(target["boxes"]) is not torch.Tensor for target in targets):
        raise ValueError("Expected all boxes in `target` to be of type torch.Tensor")
    if any(type(target["labels"]) is not torch.Tensor for target in targets):
        raise ValueError("Expected all labels in `target` to be of type torch.Tensor")

    for i, item in enumerate(targets):
        if item["boxes"].size(0) != item["labels"].size(0):
            raise ValueError(
                f"Input boxes and labels of sample {i} in targets have a"
                f" different length (expected {item['boxes'].size(0)} labels, got {item['labels'].size(0)})"
            )
    for i, item in enumerate(preds):
        if item["boxes"].size(0) != item["labels"].size(0) != item["scores"].size(0):
            raise ValueError(
                f"Input boxes, labels and scores of sample {i} in preds have a"
                f" different length (expected {item['boxes'].size(0)} labels and scores,"
                f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})"
            )


class MAP(Metric):
    r"""
    Computes the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)\
    <https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173>`_\
    for object detection predictions.
    Optionally, the mAP and mAR values can be calculated per class.

    Predicted boxes and targets have to be in Pascal VOC format
    (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    See the :meth:`update` method for more information about the input format to this metric.

    For an example on how to use this metric check the `torchmetrics examples\
    <https://github.com/PyTorchLightning/metrics/blob/master/tm_examples/detection_map.py>`_

    .. note::
        This metric is a wrapper for the
        `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        which is a standard implementation for the mAP metric for object detection. Using this metric
        therefore requires you to have `pycocotools` installed. Please install with ``pip install pycocotools`` or
        ``pip install torchmetrics[detection]``.

    .. note::
        This metric requires you to have `torchvision` version 0.8.0 or newer installed (with corresponding
        version 1.7.0 of torch or newer). Please install with ``pip install torchvision`` or
        ``pip install torchmetrics[detection]``.

    .. note::
        As the pycocotools library cannot deal with tensors directly, all results have to be transfered
        to the CPU, this might have an performance impact on your training.

    Args:
        num_classes:
            Number of classes to use for evaluation.
        iou_thresholds:
            IoU thresholds for evaluation.
        max_detection_thresholds:
            Thresholds on max detections per image.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact. default: False
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Raises:
        ImportError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ValueError:
            If ``class_metrics`` is not a boolean
    """

    def __init__(
        self,
        in_fmt: str = "xyxy",
        num_classes: Optional[int] = None,
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[float]] = None,
        class_metrics: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:  # type: ignore
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8):
            raise ImportError(
                "`MAP` metric requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision` or `pip install torchmetrics[detection]`"
            )

        self.in_fmt = in_fmt
        self.num_classes = num_classes
        self.iou_thresholds = torch.Tensor(
            iou_thresholds or torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1)
        )
        self.rec_thresholds = torch.Tensor(
            rec_thresholds or torch.linspace(0.0, 1.00, int(round((1.00 - 0.0) / 0.01)) + 1)
        )
        self.max_detection_thresholds = torch.IntTensor(max_detection_thresholds or [1, 10, 100])
        self.max_detection_thresholds, _ = torch.sort(self.max_detection_thresholds)
        self.object_area_ranges = {
            "all": [0 ** 2, 1e5 ** 2],
            "small": [0 ** 2, 32 ** 2],
            "medium": [32 ** 2, 96 ** 2],
            "large": [96 ** 2, 1e5 ** 2],
        }

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and groundtruth to the metric.

        Args:
            preds: A list consisting of dictionaries each containing the key-values\
            (each dictionary corresponds to a single image):
            - ``boxes``: torch.FloatTensor of shape
                [num_boxes, 4] containing `num_boxes` detection boxes of the format
                [xmin, ymin, xmax, ymax] in absolute image coordinates.
            - ``scores``: torch.FloatTensor of shape
                [num_boxes] containing detection scores for the boxes.
            - ``labels``: torch.IntTensor of shape
                [num_boxes] containing 0-indexed detection classes for the boxes.

            target: A list consisting of dictionaries each containing the key-values\
            (each dictionary corresponds to a single image):
            - ``boxes``: torch.FloatTensor of shape
                [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                [xmin, ymin, xmax, ymax] in absolute image coordinates.
            - ``labels``: torch.IntTensor of shape
                [num_boxes] containing 1-indexed groundtruth classes for the boxes.

        Raises:
            ValueError:
                If ``preds`` is not of type List[Dict[str, torch.Tensor]]
            ValueError:
                If ``target`` is not of type List[Dict[str, torch.Tensor]]
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores``
                and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        _input_validator(preds, target)

        for item in preds:
            self.detection_boxes.append(
                box_convert(item["boxes"], in_fmt=self.in_fmt, out_fmt="xyxy")
                if item["boxes"].size() == torch.Size([1, 4])
                else item["boxes"]
            )
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            self.groundtruth_boxes.append(
                box_convert(item["boxes"], in_fmt=self.in_fmt, out_fmt="xyxy")
                if item["boxes"].size() == torch.Size([1, 4])
                else item["boxes"]
            )
            self.groundtruth_labels.append(item["labels"])

    def _get_classes(self) -> List:
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return torch.cat(self.detection_labels + self.groundtruth_labels).unique().tolist()
        else:
            return []

    def _compute_iou(self, id: int, class_id: int, max_det: int) -> Tensor:
        gt = self.groundtruth_boxes[id]
        dt = self.detection_boxes[id]
        gt = gt[self.groundtruth_labels[id] == class_id]
        dt = dt[self.detection_labels[id] == class_id]
        if len(gt) == 0 and len(dt) == 0:
            return torch.tensor([])

        # Sort by scores and use only max detections
        scores = self.detection_scores[id]
        filtered_scores = scores[self.detection_labels[id] == class_id]
        inds = torch.argsort(filtered_scores, descending=True)
        dt = dt[inds]
        if len(dt) > max_det:
            dt = dt[:max_det]

        # generalized_box_iou
        ious = box_iou(dt, gt)
        return ious

    def _evaluate_image(self, id: int, class_id: int, area_range: List[int], max_det: int, ious: Tensor) -> Dict:
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        gt = self.groundtruth_boxes[id]
        dt = self.detection_boxes[id]
        gt = gt[self.groundtruth_labels[id] == class_id]
        dt = dt[self.detection_labels[id] == class_id]
        if len(gt) == 0 and len(dt) == 0:
            return None

        areas = box_area(gt)
        ignore_area = (areas < area_range[0]) | (areas > area_range[1])

        # sort dt highest score first, sort gt ignore last
        gtind = torch.argsort(ignore_area)
        gt = gt[gtind]
        scores = self.detection_scores[id]
        filtered_scores = scores[self.detection_labels[id] == class_id]
        dtind = torch.argsort(filtered_scores, descending=True)
        dt = dt[dtind]
        if len(dt) > max_det:
            dt = dt[:max_det]
        # load computed ious
        ious = ious[id, class_id][:, gtind] if len(ious[id, class_id]) > 0 else ious[id, class_id]

        T = len(self.iou_thresholds)
        G = len(gt)
        D = len(dt)
        gtm = torch.zeros((T, G))
        dtm = torch.zeros((T, D))
        gtIg = ignore_area
        dtIg = torch.zeros((T, D))
        if len(ious) > 0:
            for tind, t in enumerate(self.iou_thresholds):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and not gtIg[m] and gtIg[gind]:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue

                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = id
                    gtm[tind, m] = id
        # set unmatched detections outside of area range to ignore
        dt_areas = box_area(dt)
        dt_ignore_area = (dt_areas < area_range[0]) | (dt_areas > area_range[1])
        a = dt_ignore_area.reshape((1, len(dt)))
        dtIg = torch.logical_or(dtIg, torch.logical_and(dtm == 0, torch.repeat_interleave(a, T, 0)))
        return {
            "image_id": id,
            "category_id": class_id,
            "aRng": area_range,
            "maxDet": max_det,
            "dtIds": [id for d in dt],
            "gtIds": [id for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": filtered_scores,
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def compute(self) -> dict:
        """Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` scores. All detections added in
        the `update()` method are included.

        Note:
            Main `map` score is calculated with @[ IoU=0.50:0.95 | area=all | maxDets=100 ]

        Returns:
            dict containing

            - map: ``torch.Tensor``
            - map_50: ``torch.Tensor``
            - map_75: ``torch.Tensor``
            - map_small: ``torch.Tensor``
            - map_medium: ``torch.Tensor``
            - map_large: ``torch.Tensor``
            - mar_1: ``torch.Tensor``
            - mar_10: ``torch.Tensor``
            - mar_100: ``torch.Tensor``
            - mar_small: ``torch.Tensor``
            - mar_medium: ``torch.Tensor``
            - mar_large: ``torch.Tensor``
            - map_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
            - mar_100_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
        """
        classes = self._get_classes()
        # num_classes = int(self.num_classes or max(classes))
        # catIds = torch.arange(num_classes).tolist()
        overall, map, mar = self._calculate(classes)

        map_per_class_values: Tensor = torch.Tensor([-1])
        mar_100_per_class_values: Tensor = torch.Tensor([-1])

        # if class mode is enabled, evaluate metrics per class
        if self.class_metrics:
            map_per_class_list = []
            mar_100_per_class_list = []
            for class_id in self._get_classes():
                _, cls_map, cls_mar = self._calculate([class_id])

                map_per_class_list.append(cls_map.map)
                mar_100_per_class_list.append(cls_mar.mar_100)
            map_per_class_values = torch.Tensor(map_per_class_list)
            mar_100_per_class_values = torch.Tensor(mar_100_per_class_list)

        metrics = COCOMetricResults(
            map=map.map,
            map_50=map.map_50,
            map_75=map.map_75,
            map_small=map.map_small,
            map_medium=map.map_medium,
            map_large=map.map_large,
            mar_1=mar.mar_1,
            mar_10=mar.mar_10,
            mar_100=mar.mar_100,
            mar_small=mar.mar_small,
            mar_medium=mar.mar_medium,
            mar_large=mar.mar_large,
            map_per_class=map_per_class_values,
            mar_100_per_class=mar_100_per_class_values,
        )
        return metrics.__dict__

    def _summarize(
        self, results: Dict, ap: bool = True, iouThr: Optional[float] = None, areaRng: str = "all", maxDets: int = 100
    ) -> Tensor:
        aind = [i for i, aRng in enumerate(self.object_area_ranges.keys()) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(self.max_detection_thresholds) if mDet == maxDets]
        if ap:
            # dimension of precision: [TxRxKxAxM]
            s = results["precision"]
            # IoU
            if iouThr is not None:
                t = torch.where(iouThr == self.iou_thresholds)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = results["recall"]
            if iouThr is not None:
                t = torch.where(iouThr == self.iou_thresholds)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = torch.Tensor([-1])
        else:
            mean_s = torch.mean(s[s > -1])

        return mean_s

    def _calculate(self, classes) -> Tuple[Dict, MAPMetricResults, MARMetricResults]:
        img_ids = torch.arange(len(self.groundtruth_boxes), dtype=torch.int).tolist()
        imgIds = img_ids
        catIds = classes

        maxDets = self.max_detection_thresholds
        maxDet = self.max_detection_thresholds[-1]
        areaRng = self.object_area_ranges.values()

        ious = {(imgId, catId): self._compute_iou(imgId, catId, maxDet) for imgId in imgIds for catId in catIds}

        evalImgs = [
            self._evaluate_image(imgId, catId, area, maxDet, ious)
            for catId in catIds
            for area in areaRng
            for imgId in imgIds
        ]

        T = len(self.iou_thresholds)
        R = len(self.rec_thresholds)
        K = len(catIds)
        A = len(self.object_area_ranges)
        M = len(self.max_detection_thresholds)
        precision = -torch.ones((T, R, K, A, M))
        recall = -torch.ones((T, K, A, M))
        scores = -torch.ones((T, R, K, A, M))

        setK = set(catIds)
        setA = set(map(tuple, self.object_area_ranges.values()))
        setM = set(self.max_detection_thresholds.tolist())
        setI = set(imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(catIds) if k in setK]
        m_list = [m for n, m in enumerate(maxDets.tolist()) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(imgIds) if i in setI]
        I0 = len(imgIds)
        A0 = len(areaRng)
        # # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = torch.cat([e["dtScores"][:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = torch.argsort(dtScores, descending=True)
                    dtScoresSorted = dtScores[inds]

                    dtm = torch.cat([e["dtMatches"][:, :maxDet] for e in E], axis=1)[:, inds]
                    dtIg = torch.cat([e["dtIgnore"][:, :maxDet] for e in E], axis=1)[:, inds]
                    gtIg = torch.cat([e["gtIgnore"] for e in E])
                    npig = torch.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = torch.logical_and(dtm, torch.logical_not(dtIg))
                    fps = torch.logical_and(torch.logical_not(dtm), torch.logical_not(dtIg))

                    tp_sum = torch.cumsum(tps, axis=1, dtype=torch.float)
                    fp_sum = torch.cumsum(fps, axis=1, dtype=torch.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
                        q = torch.zeros((R,))
                        ss = torch.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # TODO: optimize
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        # TODO: optimize
                        inds = torch.searchsorted(rc, self.rec_thresholds, right=False)
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except Exception:
                            pass
                        precision[t, :, k, a, m] = torch.tensor(q)
                        scores[t, :, k, a, m] = torch.tensor(ss)

        results = {
            "counts": [T, R, K, A, M],
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        map_metrics = MAPMetricResults(
            map=self._summarize(results, True),
            map_50=self._summarize(results, True, iouThr=0.5, maxDets=self.max_detection_thresholds[2]),
            map_75=self._summarize(results, True, iouThr=0.75, maxDets=self.max_detection_thresholds[2]),
            map_small=self._summarize(results, True, areaRng="small", maxDets=self.max_detection_thresholds[2]),
            map_medium=self._summarize(results, True, areaRng="medium", maxDets=self.max_detection_thresholds[2]),
            map_large=self._summarize(results, True, areaRng="large", maxDets=self.max_detection_thresholds[2]),
        )
        mar_metrics = MARMetricResults(
            mar_1=self._summarize(results, False, maxDets=self.max_detection_thresholds[0]),
            mar_10=self._summarize(results, False, maxDets=self.max_detection_thresholds[1]),
            mar_100=self._summarize(results, False, maxDets=self.max_detection_thresholds[2]),
            mar_small=self._summarize(results, False, areaRng="small", maxDets=self.max_detection_thresholds[2]),
            mar_medium=self._summarize(results, False, areaRng="medium", maxDets=self.max_detection_thresholds[2]),
            mar_large=self._summarize(results, False, areaRng="large", maxDets=self.max_detection_thresholds[2]),
        )
        return results, map_metrics, mar_metrics
