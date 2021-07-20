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

from typing import Any, Optional, List

import torch
from torch import Tensor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchmetrics.metric import Metric


@dataclass
class MAPMetricResults():
    map_value: Tensor
    mar_value: Tensor
    map_per_class_value: List[Tensor]
    mar_per_class_value: List[Tensor]


class hide_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MAP(Metric):
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
        self.add_state('average_precision', default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx='mean')
        self.add_state('average_recall', default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx='mean')
        self.num_classes = num_classes

        for class_id in range(num_classes):
            self.add_state(f'ap_{class_id}', default=torch.tensor(data=[], dtype=torch.float),
                           dist_reduce_fx='mean')
            self.add_state(f'ar_{class_id}', default=torch.tensor(data=[], dtype=torch.float),
                           dist_reduce_fx='mean')

    def update(self, preds, target, class_names):
        coco_target = COCO()
        coco_target.dataset = get_coco_target(target=target, class_names=class_names)

        coco_preds = COCO()
        coco_preds.dataset = get_coco_preds(preds=preds, coco_target=coco_target)

        with hide_prints():
            coco_target.createIndex()
            coco_preds.createIndex()
            coco_eval = COCOeval(coco_target, coco_preds, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        self.average_precision = torch.cat(
            (self.average_precision,
             torch.tensor([stats[0]], dtype=torch.float, device=self.average_precision.device)))

        self.average_recall = torch.cat(
            (self.average_recall, torch.tensor([stats[8]], dtype=torch.float, device=self.average_recall.device)))

        for class_id in range(self.num_classes):
            coco_eval.params.catIds = [class_id]
            with hide_prints():
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats

            current_value = getattr(self, f'ap_{class_id}')
            class_map_value = torch.cat(
                (current_value, torch.tensor([stats[0]], dtype=torch.float, device=current_value.device)))
            setattr(self, f'ap_{class_id}', class_map_value)

            current_value = getattr(self, f'ar_{class_id}')
            class_map_value = torch.cat(
                (current_value, torch.tensor([stats[8]], dtype=torch.float, device=current_value.device)))
            setattr(self, f'ar_{class_id}', class_map_value)

    def compute(self):
        map_per_class_value = [torch.mean(getattr(self, f'ap_{class_id}')) for class_id in
                               range(self.num_classes)]
        mar_per_class_value = [torch.mean(getattr(self, f'ar_{class_id}')) for class_id in
                               range(self.num_classes)]
        metrics = MAPMetricResults(map_value=torch.mean(self.average_precision),
                                   mar_value=torch.mean(self.average_recall),
                                   map_per_class_value=map_per_class_value,
                                   mar_per_class_value=mar_per_class_value)
        return metrics


def get_coco_target(target, class_names):
    image_list = []
    annotation_list = []
    annotation_id = 1

    for i, (boxes, labels) in enumerate(zip(target['boxes'], target['labels'])):
        boxes = boxes.cpu().tolist()
        labels = labels.cpu().tolist()
        image_list.append({'id': i})
        for box, label in zip(boxes, labels):
            box = box
            annotation_list.append({
                'id': annotation_id,
                'image_id': i,
                'bbox': box,
                'category_id': label,
                'area': box[2] * box[3],
                'iscrowd': 0
            })
            annotation_id += 1
    classes_list = [{'id': i, 'name': val} for i, val in enumerate(class_names)]
    return {'images': image_list, 'annotations': annotation_list, 'categories': classes_list}


def get_coco_preds(preds, coco_target):
    image_list = [img for img in coco_target.dataset['images']]
    annotation_list = []
    annotation_id = 1
    for i, (boxes, labels) in enumerate(preds):
        boxes_scores = boxes.cpu().tolist()
        boxes = [box[:4] for box in boxes_scores]
        scores = [box[-1] for box in boxes_scores]
        labels = labels.cpu().tolist()
        for box, label, score in zip(boxes, labels, scores):
            box = box
            annotation_list.append({
                'id': annotation_id,
                'image_id': i,
                'category_id': label,
                'bbox': box,
                'score': score,
                'area': box[2] * box[3],
                'iscrowd': 0
            })
            annotation_id += 1

    return {'images': image_list, 'annotations': annotation_list, 'categories': coco_target.dataset['categories']}
