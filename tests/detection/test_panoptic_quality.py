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

from tests.helpers import seed_all
import pytest
import torch

from tests.helpers.testers import MetricTester
from torchmetrics.detection.panoptic_quality import PanopticQuality

seed_all(42)


def test_empty_metric():
    """Test empty metric."""
    metric = PanopticQuality(things={}, stuff={})
    metric.compute()


def test_perfect_match():
    """Test evaluation on random image."""
    metric = PanopticQuality(things={0: "person", 1: "dog", 3: "cat"}, stuff={6: "sky", 8: "grass"})
    height, width = 300, 400
    preds = torch.randint(low=0, high=9, size=(height, width, 2))
    metric.update(preds, preds)
    metric = metric.compute()
    for metric_class in ["all", "things", "stuff"]:
        assert metric[metric_class]["pq"] == 1.0
        assert metric[metric_class]["rq"] == 1.0
        assert metric[metric_class]["sq"] == 1.0


def test_error_on_wrong_input():
    """Test class input validation."""

    with pytest.raises(ValueError):
        PanopticQuality(things={"person": 0}, stuff={1: "sky"})

    with pytest.raises(ValueError):
        PanopticQuality(things={0: "person"}, stuff={"sky": 1})

    with pytest.raises(ValueError):
        PanopticQuality(things={0: "person"}, stuff={0: "sky"})

    metric = PanopticQuality(things={0: "person", 1: "dog", 3: "cat"}, stuff={6: "sky", 8: "grass"})
    valid_image = torch.randint(low=0, high=9, size=(400, 300, 2))
    metric.update(valid_image, valid_image)

    with pytest.raises(ValueError):
        metric.update([], valid_image)  # type: ignore

    with pytest.raises(ValueError):
        metric.update(valid_image, [])  # type: ignore

    with pytest.raises(ValueError):
        preds = torch.randint(low=0, high=9, size=(400, 300, 2))
        target = torch.randint(low=0, high=9, size=(300, 400, 2))
        metric.update(preds, target)

    with pytest.raises(ValueError):
        preds = torch.randint(low=0, high=9, size=(400, 300))
        metric.update(preds, preds)
