# Copyright The Lightning team.
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
import pytest
import torch
from torch.nn.functional import one_hot

from torchmetrics.functional.segmentation import dice_score, mean_iou


@pytest.mark.parametrize("metric", [dice_score, mean_iou])
@pytest.mark.parametrize("scores_are_float", [False, True])
@pytest.mark.parametrize("input_format", ["one-hot", "mixed"])
@pytest.mark.parametrize("num_classes", [2, 4])
def test_segmentation_class_count(metric, scores_are_float, input_format, num_classes):
    """Class count must match the class axis in one-hot inputs."""
    labels = torch.tensor([[[0, 1, 2], [2, 0, 1]]])
    preds = one_hot(labels, num_classes=3).movedim(-1, 1)
    if scores_are_float:
        preds = preds.float()
    target = labels if input_format == "mixed" else preds

    with pytest.raises(ValueError, match="num_classes"):
        metric(preds, target, num_classes=num_classes, input_format=input_format)

    torch.testing.assert_close(
        metric(preds, target, num_classes=3, input_format=input_format),
        torch.tensor([1.0]),
    )
