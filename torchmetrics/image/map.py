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
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _PYCOCOTOOLS_AVAILABLE

if _PYCOCOTOOLS_AVAILABLE:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
else:
    COCO, COCOeval = None, None

COCO_STATS_MAP_VALUE_INDEX = 0
COCO_STATS_MAR_VALUE_INDEX = 8


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""

    map_value: Tensor
    mar_value: Tensor
    map_per_class_value: List[Tensor]
    mar_per_class_value: List[Tensor]


class _hide_prints:
    """Internal helper context to suppress the default output of the pycocotools package."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MAP(Metric):
    """Computes the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) for object detection predictions.
    Optionally, the mAP and mAR values can be calculated per class.

    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    Boxes and targets have to be in COCO format with the box score at the end
    (x-top left, y-top left, width, height, score)

    .. note::
        This metric is a wrapper for the `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_, which is a standard implementation for the mAP metric for object detection. Using this metric
        therefore requires you to have `pycocotools` installed. Please install with `pip install pycocotools` or
        `pip install torchmetrics[image]`


    .. note::
        As the pycocotools library cannot deal with tensors directly, all results have to be transfered
        to the CPU, this might have an performance impact on your training

    Args:
        num_classes:
            Number of classes, required for mAP values per class. default: 0 (deactivate)
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Raises:
        RuntimeError:
            If ``pycocotools`` is not installed
        ValueError:
            If ``num_classes`` is not an integer larger or equal to 0
    """

    def __init__(
        self,
        num_classes: int = 0,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if not _PYCOCOTOOLS_AVAILABLE:
            raise RuntimeError(
                "MAP metric requires pycocotools are installed"
                "Please install as `pip install pycocotools` or `pip install torchmetrics[image]`"
            )
        self.add_state("average_precision", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
        self.add_state("average_recall", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
        self.num_classes = num_classes

        for class_id in range(num_classes):
            self.add_state(f"ap_{class_id}", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
            self.add_state(f"ar_{class_id}", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")

    def update(self, preds: list, target: list):
        if len(preds[0]) != len(target[0]):
            raise ValueError("preds and targets need to be of the same length")

        coco_target, coco_preds = COCO(), COCO()

        coco_target.dataset = self.get_coco_format(input=target)
        coco_preds.dataset = self.get_coco_format(input=preds, is_pred=True)

        with _hide_prints():
            coco_target.createIndex()
            coco_preds.createIndex()
            coco_eval = COCOeval(coco_target, coco_preds, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        self.average_precision = torch.cat(
            (
                self.average_precision,
                torch.tensor(
                    [stats[COCO_STATS_MAP_VALUE_INDEX]], dtype=torch.float, device=self.average_precision.device
                ),
            )
        )

        self.average_recall = torch.cat(
            (
                self.average_recall,
                torch.tensor([stats[COCO_STATS_MAR_VALUE_INDEX]], dtype=torch.float, device=self.average_recall.device),
            )
        )

        # if class mode is enabled, evaluate metrics per class
        for class_id in range(self.num_classes):
            coco_eval.params.catIds = [class_id]
            with _hide_prints():
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats

            current_value = getattr(self, f"ap_{class_id}")
            class_map_value = torch.cat(
                (
                    current_value,
                    torch.tensor([stats[COCO_STATS_MAP_VALUE_INDEX]], dtype=torch.float, device=current_value.device),
                )
            )
            setattr(self, f"ap_{class_id}", class_map_value)

            current_value = getattr(self, f"ar_{class_id}")
            class_map_value = torch.cat(
                (
                    current_value,
                    torch.tensor([stats[COCO_STATS_MAR_VALUE_INDEX]], dtype=torch.float, device=current_value.device),
                )
            )
            setattr(self, f"ar_{class_id}", class_map_value)

    def compute(self):
        map_per_class_value = [torch.mean(getattr(self, f"ap_{class_id}")) for class_id in range(self.num_classes)]
        mar_per_class_value = [torch.mean(getattr(self, f"ar_{class_id}")) for class_id in range(self.num_classes)]
        metrics = MAPMetricResults(
            map_value=torch.mean(self.average_precision),
            mar_value=torch.mean(self.average_recall),
            map_per_class_value=map_per_class_value,
            mar_per_class_value=mar_per_class_value,
        )
        return metrics

    def get_coco_format(self, input: List, is_pred: bool = False) -> Dict:
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for i, (boxes, labels) in enumerate(
            zip(input[0], input[1])
        ):  # TODO, what is the default bounding box / label format
            boxes = boxes.cpu().tolist()
            labels = labels.cpu().tolist()
            images.append({"id": i})
            for box, label in zip(boxes, labels):
                annotation = {
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": label,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
                if is_pred:
                    annotation["bbox"] = box[:4]
                    annotation["score"] = box[-1]
                else:
                    annotation["bbox"] = box
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in range(self.num_classes)]
        return {"images": images, "annotations": annotations, "categories": classes}
