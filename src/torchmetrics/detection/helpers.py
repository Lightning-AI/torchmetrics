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
from typing import Dict, Sequence

from torch import Tensor


def _input_validator(
    preds: Sequence[Dict[str, Tensor]], targets: Sequence[Dict[str, Tensor]], iou_type: str = "bbox"
) -> None:
    """Ensure the correct input format of `preds` and `targets`."""
    if iou_type == "bbox":
        item_val_name = "boxes"
    elif iou_type == "segm":
        item_val_name = "masks"
    else:
        raise Exception(f"IOU type {iou_type} is not supported")

    if not isinstance(preds, Sequence):
        raise ValueError(f"Expected argument `preds` to be of type Sequence, but got {preds}")
    if not isinstance(targets, Sequence):
        raise ValueError(f"Expected argument `target` to be of type Sequence, but got {targets}")
    if len(preds) != len(targets):
        raise ValueError(
            f"Expected argument `preds` and `target` to have the same length, but got {len(preds)} and {len(targets)}"
        )

    for k in [item_val_name, "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in [item_val_name, "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(pred[item_val_name]) is not Tensor for pred in preds):
        raise ValueError(f"Expected all {item_val_name} in `preds` to be of type Tensor")
    if any(type(pred["scores"]) is not Tensor for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type Tensor")
    if any(type(pred["labels"]) is not Tensor for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type Tensor")
    if any(type(target[item_val_name]) is not Tensor for target in targets):
        raise ValueError(f"Expected all {item_val_name} in `target` to be of type Tensor")
    if any(type(target["labels"]) is not Tensor for target in targets):
        raise ValueError("Expected all labels in `target` to be of type Tensor")

    for i, item in enumerate(targets):
        if item[item_val_name].size(0) != item["labels"].size(0):
            raise ValueError(
                f"Input {item_val_name} and labels of sample {i} in targets have a"
                f" different length (expected {item[item_val_name].size(0)} labels, got {item['labels'].size(0)})"
            )
    for i, item in enumerate(preds):
        if not (item[item_val_name].size(0) == item["labels"].size(0) == item["scores"].size(0)):
            raise ValueError(
                f"Input {item_val_name}, labels and scores of sample {i} in predictions have a"
                f" different length (expected {item[item_val_name].size(0)} labels and scores,"
                f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})"
            )


def _fix_empty_tensors(boxes: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if boxes.numel() == 0 and boxes.ndim == 1:
        return boxes.unsqueeze(0)
    return boxes
