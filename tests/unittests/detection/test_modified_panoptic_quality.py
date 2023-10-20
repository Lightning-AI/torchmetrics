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
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torchmetrics.detection import ModifiedPanopticQuality
from torchmetrics.functional.detection import modified_panoptic_quality

from unittests import _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


_INPUTS_0 = _Input(
    # Shape of input tensors is (num_batches, batch_size, height, width, 2).
    preds=torch.tensor(
        [
            [[6, 0], [0, 0], [6, 0], [6, 0], [0, 1]],
            [[0, 0], [0, 0], [6, 0], [0, 1], [0, 1]],
            [[0, 0], [0, 0], [6, 0], [0, 1], [1, 0]],
            [[0, 0], [7, 0], [6, 0], [1, 0], [1, 0]],
            [[0, 0], [7, 0], [7, 0], [7, 0], [7, 0]],
        ]
    )
    .reshape((1, 1, 5, 5, 2))
    .repeat(2, 1, 1, 1, 1),
    target=torch.tensor(
        [
            [[6, 0], [6, 0], [6, 0], [6, 0], [0, 0]],
            [[0, 1], [0, 1], [6, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 1], [6, 0], [1, 0], [1, 0]],
            [[0, 1], [7, 0], [7, 0], [1, 0], [1, 0]],
            [[0, 1], [7, 0], [7, 0], [7, 0], [7, 0]],
        ]
    )
    .reshape((1, 1, 5, 5, 2))
    .repeat(2, 1, 1, 1, 1),
)
_INPUTS_1 = _Input(
    # Shape of input tensors is (num_batches, batch_size, num_points, 2).
    # NOTE: IoU for stuff category 6 is < 0.5, modified PQ behaves differently there.
    preds=torch.tensor([[0, 0], [0, 1], [6, 0], [7, 0], [0, 2], [1, 0]]).reshape((1, 1, 6, 2)).repeat(2, 1, 1, 1),
    target=torch.tensor([[0, 1], [0, 0], [6, 0], [7, 0], [6, 0], [255, 0]]).reshape((1, 1, 6, 2)).repeat(2, 1, 1, 1),
)
_ARGS_0 = {"things": {0, 1}, "stuffs": {6, 7}}
_ARGS_1 = {"things": {2}, "stuffs": {3}, "allow_unknown_preds_category": True}
_ARGS_2 = {"things": {0, 1}, "stuffs": {6, 7}}

# TODO: Improve _compare_fn by calling https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
# directly and compare at runtime on multiple examples.


def _compare_fn_0_0(preds, target) -> np.ndarray:
    """Baseline result for the _INPUTS_0, _ARGS_0 combination."""
    return np.array([0.7753])


def _compare_fn_0_1(preds, target) -> np.ndarray:
    """Baseline result for the _INPUTS_0, _ARGS_1 combination."""
    return np.array([np.nan])


def _compare_fn_1_2(preds, target) -> np.ndarray:
    """Baseline result for the _INPUTS_1, _ARGS_2 combination."""
    return np.array([23 / 30])


class TestModifiedPanopticQuality(MetricTester):
    """Test class for `ModifiedPanopticQuality` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize(
        ("inputs", "args", "reference_metric"),
        [
            (_INPUTS_0, _ARGS_0, _compare_fn_0_0),
            (_INPUTS_0, _ARGS_1, _compare_fn_0_1),
            (_INPUTS_1, _ARGS_2, _compare_fn_1_2),
        ],
    )
    def test_panoptic_quality_class(self, ddp, inputs, args, reference_metric):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=inputs.preds,
            target=inputs.target,
            metric_class=ModifiedPanopticQuality,
            reference_metric=reference_metric,
            check_batch=False,
            metric_args=args,
        )

    def test_panoptic_quality_functional(self):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            _INPUTS_0.preds,
            _INPUTS_0.target,
            metric_functional=modified_panoptic_quality,
            reference_metric=_compare_fn_0_0,
            metric_args=_ARGS_0,
        )


def test_empty_metric():
    """Test empty metric."""
    with pytest.raises(ValueError, match="At least one of `things` and `stuffs` must be non-empty"):
        metric = ModifiedPanopticQuality(things=[], stuffs=[])

    metric = ModifiedPanopticQuality(things=[0], stuffs=[])
    assert torch.isnan(metric.compute())


def test_error_on_wrong_input():
    """Test class input validation."""
    with pytest.raises(TypeError, match="Expected argument `stuffs` to contain `int` categories.*"):
        ModifiedPanopticQuality(things={0}, stuffs={"sky"})

    with pytest.raises(ValueError, match="Expected arguments `things` and `stuffs` to have distinct keys.*"):
        ModifiedPanopticQuality(things={0}, stuffs={0})

    metric = ModifiedPanopticQuality(things={0, 1, 3}, stuffs={2, 8}, allow_unknown_preds_category=True)
    valid_images = torch.randint(low=0, high=9, size=(8, 64, 64, 2))
    metric.update(valid_images, valid_images)
    valid_point_clouds = torch.randint(low=0, high=9, size=(1, 100, 2))
    metric.update(valid_point_clouds, valid_point_clouds)

    with pytest.raises(TypeError, match="Expected argument `preds` to be of type `torch.Tensor`.*"):
        metric.update([], valid_images)

    with pytest.raises(TypeError, match="Expected argument `target` to be of type `torch.Tensor`.*"):
        metric.update(valid_images, [])

    preds = torch.randint(low=0, high=9, size=(2, 400, 300, 2))
    target = torch.randint(low=0, high=9, size=(2, 30, 40, 2))
    with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have the same shape.*"):
        metric.update(preds, target)

    preds = torch.randint(low=0, high=9, size=(1, 2))
    with pytest.raises(ValueError, match="Expected argument `preds` to have at least one spatial dimension.*"):
        metric.update(preds, preds)

    preds = torch.randint(low=0, high=9, size=(1, 64, 64, 8))
    with pytest.raises(
        ValueError, match="Expected argument `preds` to have exactly 2 channels in the last dimension.*"
    ):
        metric.update(preds, preds)

    metric = ModifiedPanopticQuality(things=[0], stuffs=[1], allow_unknown_preds_category=False)
    preds = torch.randint(low=0, high=1, size=(1, 100, 2))
    preds[0, 0, 0] = 2
    with pytest.raises(ValueError, match="Unknown categories found.*"):
        metric.update(preds, preds)


def test_extreme_values():
    """Test that the metric returns expected values in trivial cases."""
    # Exact match between preds and target => metric is 1
    assert modified_panoptic_quality(_INPUTS_0.target[0], _INPUTS_0.target[0], **_ARGS_0) == 1.0
    # Every element of the prediction is wrong => metric is 0
    assert modified_panoptic_quality(_INPUTS_0.target[0], _INPUTS_0.target[0] + 1, **_ARGS_0) == 0.0


@pytest.mark.parametrize(
    ("inputs", "args", "cat_dim"),
    [
        (_INPUTS_0, _ARGS_0, 0),
        (_INPUTS_0, _ARGS_0, 1),
        (_INPUTS_0, _ARGS_0, 2),
        (_INPUTS_1, _ARGS_2, 0),
        (_INPUTS_1, _ARGS_2, 1),
    ],
)
def test_ignore_mask(inputs: _Input, args: Dict[str, Any], cat_dim: int):
    """Test that the metric correctly ignores regions of the inputs that do not map to a know category ID."""
    preds = inputs.preds[0]
    target = inputs.target[0]
    value = modified_panoptic_quality(preds, target, **args)
    ignored_regions = torch.zeros_like(preds)
    ignored_regions[..., 0] = 255
    preds_new = torch.cat([preds, preds], dim=cat_dim)
    target_new = torch.cat([target, ignored_regions], dim=cat_dim)
    value_new = modified_panoptic_quality(preds_new, target_new, **args)
    assert value == value_new
