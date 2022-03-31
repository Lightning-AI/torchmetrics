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
from collections import namedtuple
from tests.helpers import seed_all
import numpy as np
import pytest
import torch

from tests.helpers.testers import MetricTester
from torchmetrics.detection.panoptic_quality import PanopticQuality
from torchmetrics.functional.detection.panoptic_quality import panoptic_quality

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

_inputs = Input(
    preds=torch.tensor(
        [
            [
                [[6, 0], [0, 0], [6, 0], [6, 0], [0, 1]],
                [[0, 0], [0, 0], [6, 0], [0, 1], [0, 1]],
                [[0, 0], [0, 0], [6, 0], [0, 1], [1, 0]],
                [[0, 0], [7, 0], [6, 0], [1, 0], [1, 0]],
                [[0, 0], [7, 0], [7, 0], [7, 0], [7, 0]],
            ]
        ]
    ),
    target=torch.tensor(
        [
            [
                [[6, 0], [6, 0], [6, 0], [6, 0], [0, 0]],
                [[0, 1], [0, 1], [6, 0], [0, 0], [0, 0]],
                [[0, 1], [0, 1], [6, 0], [1, 0], [1, 0]],
                [[0, 1], [7, 0], [7, 0], [1, 0], [1, 0]],
                [[0, 1], [7, 0], [7, 0], [7, 0], [7, 0]],
            ]
        ]
    ),
)
_args = dict(things={0: "person", 1: "cat"}, stuff={6: "sky", 7: "grass"})


def _compare_fn(preds, target) -> dict:
    return np.array([0.7753])


class TestPanopticQuality(MetricTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_panoptic_quality_class(self, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs.preds,
            target=_inputs.target,
            metric_class=PanopticQuality,
            sk_metric=_compare_fn,
            dist_sync_on_step=dist_sync_on_step,
            check_batch=False,
            metric_args=_args,
        )

    def test_panoptic_quality_fn(self):
        self.run_functional_metric_test(
            _inputs.preds,
            _inputs.target,
            metric_functional=panoptic_quality,
            sk_metric=_compare_fn,
            metric_args=_args,
        )


def test_empty_metric():
    """Test empty metric."""
    metric = PanopticQuality(things={}, stuff={})
    metric.compute()


def test_error_on_wrong_input():
    """Test class input validation."""

    with pytest.raises(ValueError):
        PanopticQuality(things={"person": 0}, stuff={1: "sky"})

    with pytest.raises(ValueError):
        PanopticQuality(things={0: "person"}, stuff={"sky": 1})

    with pytest.raises(ValueError):
        PanopticQuality(things={0: "person"}, stuff={0: "sky"})

    metric = PanopticQuality(
        things={0: "person", 1: "dog", 3: "cat"}, stuff={6: "sky", 8: "grass"}, allow_unknown_preds_category=True
    )
    valid_image = torch.randint(low=0, high=9, size=(400, 300, 2))
    metric.update(valid_image, valid_image)

    with pytest.raises(ValueError):
        metric.update([], valid_image)  # type: ignore

    with pytest.raises(ValueError):
        metric.update(valid_image, [])  # type: ignore

    with pytest.raises(ValueError):
        preds = torch.randint(low=0, high=9, size=(400, 300, 2))
        target = torch.randint(low=0, high=9, size=(30, 40, 2))
        metric.update(preds, target)

    with pytest.raises(ValueError):
        preds = torch.randint(low=0, high=9, size=(400, 300))
        metric.update(preds, preds)


if __name__ == "__main__":
    test = TestPanopticQuality()
    test.test_panoptic_quality_class(False, False)
