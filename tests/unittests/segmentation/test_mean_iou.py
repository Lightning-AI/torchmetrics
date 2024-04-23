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

from functools import partial

import pytest
import torch
from monai.metrics.meaniou import compute_iou
from torchmetrics.functional.segmentation.mean_iou import mean_iou
from torchmetrics.segmentation.mean_iou import MeanIoU

from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, _Input
from unittests._helpers.testers import MetricTester

_inputs1 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16)),
)
_inputs2 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)),
)
_inputs3 = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
)


def _reference_mean_iou(
    preds: torch.Tensor,
    target: torch.Tensor,
    include_background: bool = True,
    per_class: bool = True,
    reduce: bool = True,
):
    """Calculate reference metric for `MeanIoU`."""
    if (preds.bool() != preds).any():  # preds is an index tensor
        preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
    if (target.bool() != target).any():  # target is an index tensor
        target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)

    val = compute_iou(preds, target, include_background=include_background)
    val[torch.isnan(val)] = 0.0
    if reduce:
        return torch.mean(val, 0) if per_class else torch.mean(val)
    return val


@pytest.mark.parametrize(
    "preds, target",
    [
        (_inputs1.preds, _inputs1.target),
        (_inputs2.preds, _inputs2.target),
        (_inputs3.preds, _inputs3.target),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
class TestMeanIoU(MetricTester):
    """Test class for `MeanIoU` metric."""

    atol = 1e-4

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("per_class", [True, False])
    def test_mean_iou_class(self, preds, target, include_background, per_class, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MeanIoU,
            reference_metric=partial(
                _reference_mean_iou, include_background=include_background, per_class=per_class, reduce=True
            ),
            metric_args={"num_classes": NUM_CLASSES, "include_background": include_background, "per_class": per_class},
        )

    def test_mean_iou_functional(self, preds, target, include_background):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=mean_iou,
            reference_metric=partial(_reference_mean_iou, include_background=include_background, reduce=False),
            metric_args={"num_classes": NUM_CLASSES, "include_background": include_background, "per_class": True},
        )
