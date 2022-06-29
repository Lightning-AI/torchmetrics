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
"""An example of how to the predictions and target should be defined for the MAP object detection metric To run:

python detection_map.py.
"""

from torch import BoolTensor, IntTensor, Tensor

from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Preds should be a list of elements, where each element is a dict
# containing 3 keys: boxes, scores, labels
mask_pred = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
preds = [
    dict(
        # The boxes keyword should contain an [N,4] tensor,
        # where N is the number of detected boxes with boxes of the format
        # [xmin, ymin, xmax, ymax] in absolute image coordinates
        boxes=Tensor([[258.0, 41.0, 606.0, 285.0]]),
        # The scores keyword should contain an [N,] tensor where
        # each element is confidence score between 0 and 1
        scores=Tensor([0.536]),
        # The labels keyword should contain an [N,] tensor
        # with integers of the predicted classes
        labels=IntTensor([0]),
        # The masks keyword should contain an [N,H,W] tensor,
        # where H and W are the image height and width, respectively,
        # with boolean masks. This is only required when iou_type is `segm`.
        masks=BoolTensor([mask_pred]),
    )
]

# Target should be a list of elements, where each element is a dict
# containing 2 keys: boxes and labels (and masks, if iou_type is `segm`).
# Each keyword should be formatted similar to the preds argument.
# The number of elements in preds and target need to match
mask_tgt = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
]
target = [
    dict(
        boxes=Tensor([[214.0, 41.0, 562.0, 285.0]]),
        labels=IntTensor([0]),
        masks=BoolTensor([mask_tgt]),
    )
]

if __name__ == "__main__":
    # Initialize metric
    metric = MeanAveragePrecision(iou_type="bbox")

    # Update metric with predictions and respective ground truth
    metric.update(preds, target)

    # Compute the results
    result = metric.compute()
    print(result)
