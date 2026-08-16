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

from torchmetrics.detection.mean_ap_3d import MeanAveragePrecision3D
from torchmetrics.functional.detection.map_3d import _pairwise_iou_3d, mean_average_precision_3d


def test_pairwise_iou_3d():
    """Check the pairwise 3D IoU computation against hand-derived values."""
    boxes1 = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
    boxes2 = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
        [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
    ])
    iou = _pairwise_iou_3d(boxes1, boxes2)
    # identical boxes -> IoU 1.0
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))
    # 1x1x1 intersection out of a union of 8 + 8 - 1 = 15
    assert torch.isclose(iou[0, 1], torch.tensor(1 / 15))
    # disjoint boxes -> IoU 0.0
    assert torch.isclose(iou[0, 2], torch.tensor(0.0))


def test_map_3d_perfect_match():
    """A prediction identical to the target should give AP/AR of 1 for all thresholds."""
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    target = [{"boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]), "labels": torch.tensor([0])}]

    result = mean_average_precision_3d(preds, target)
    assert torch.isclose(result["map"], torch.tensor(1.0))
    assert torch.isclose(result["map_50"], torch.tensor(1.0))
    assert torch.isclose(result["map_75"], torch.tensor(1.0))
    assert torch.isclose(result["mar_100"], torch.tensor(1.0))


def test_map_3d_no_overlap():
    """A prediction that does not overlap the target should give AP/AR of 0."""
    preds = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 10.0, 12.0, 12.0, 12.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    target = [{"boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]), "labels": torch.tensor([0])}]

    result = mean_average_precision_3d(preds, target)
    assert torch.isclose(result["map"], torch.tensor(0.0))
    assert torch.isclose(result["mar_100"], torch.tensor(0.0))


def test_map_3d_box_format():
    """The ``xyzxyz`` and ``xyzwhd`` box formats should agree on the same underlying box."""
    preds_whd = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    target_whd = [{"boxes": torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]), "labels": torch.tensor([0])}]
    preds_xyz = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    target_xyz = [{"boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]), "labels": torch.tensor([0])}]

    result_whd = mean_average_precision_3d(preds_whd, target_whd, box_format="xyzwhd")
    result_xyz = mean_average_precision_3d(preds_xyz, target_xyz, box_format="xyzxyz")
    assert torch.isclose(result_whd["map"], result_xyz["map"])


def test_map_3d_class_metrics():
    """Per-class metrics should be returned when ``class_metrics=True`` and match the observed classes."""
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [10.0, 10.0, 10.0, 12.0, 12.0, 12.0]]),
            "scores": torch.tensor([0.9, 0.1]),
            "labels": torch.tensor([0, 1]),
        }
    ]
    target = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [10.0, 10.0, 10.0, 12.0, 12.0, 12.0]]),
            "labels": torch.tensor([0, 1]),
        }
    ]

    result = mean_average_precision_3d(preds, target, class_metrics=True)
    assert torch.equal(result["classes"], torch.tensor([0, 1], dtype=torch.int32))
    assert result["map_per_class"].shape == (2,)
    assert torch.allclose(result["map_per_class"], torch.tensor([1.0, 1.0]))


def test_map_3d_no_predictions():
    """If there are no predictions for an image with ground truth, mAP/mAR should be 0."""
    preds = [
        {
            "boxes": torch.zeros((0, 6)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    ]
    target = [{"boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]), "labels": torch.tensor([0])}]

    result = mean_average_precision_3d(preds, target)
    assert torch.isclose(result["map"], torch.tensor(0.0))


def test_map_3d_no_ground_truth():
    """If there is no ground truth at all, the metric should not error and report -1."""
    preds = [
        {
            "boxes": torch.zeros((0, 6)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    ]
    target = [{"boxes": torch.zeros((0, 6)), "labels": torch.zeros((0,), dtype=torch.long)}]

    result = mean_average_precision_3d(preds, target)
    assert result["map"] == -1


def test_module_matches_functional():
    """The module interface should match the functional interface for the same inputs."""
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        },
        {
            "boxes": torch.tensor([[5.0, 5.0, 5.0, 1.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.4]),
            "labels": torch.tensor([0]),
        },
    ]
    target = [
        {"boxes": torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[5.0, 5.0, 5.0, 1.0, 1.0, 1.0]]), "labels": torch.tensor([0])},
    ]

    functional_result = mean_average_precision_3d(preds, target)

    metric = MeanAveragePrecision3D()
    metric.update(preds, target)
    module_result = metric.compute()

    for key in functional_result:
        assert torch.equal(torch.as_tensor(functional_result[key]), torch.as_tensor(module_result[key]))


@pytest.mark.parametrize("box_format", ["xyzwhd", "xyzxyz"])
def test_invalid_box_format_raises(box_format):
    """Sanity check that the module accepts both supported box formats without error."""
    MeanAveragePrecision3D(box_format=box_format)


def test_invalid_box_format_raises_value_error():
    """An unsupported ``box_format`` should raise a ``ValueError``."""
    with pytest.raises(ValueError, match=r"Expected argument `box_format` to be one of.*"):
        MeanAveragePrecision3D(box_format="invalid")
