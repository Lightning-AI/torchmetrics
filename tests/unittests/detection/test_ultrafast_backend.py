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
"""Complete-output and lifecycle checks for the optional ultrafast COCO backend."""

import importlib
import pickle
from copy import deepcopy

import pytest
import torch

from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend
from torchmetrics.functional.detection import mean_average_precision
from torchmetrics.utilities.imports import _ULTRAFAST_COCO_AVAILABLE

pytestmark = pytest.mark.skipif(not _ULTRAFAST_COCO_AVAILABLE, reason="requires ultrafast-pycocotools>=0.1.11")


def _inputs():
    boxes = torch.tensor([[1.0, 1.0, 6.0, 6.0], [2.0, 1.0, 7.0, 6.0], [8.0, 8.0, 12.0, 12.0]])
    masks = torch.zeros((3, 16, 16), dtype=torch.bool)
    for mask, box in zip(masks, boxes.int()):
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True
    pred = {"boxes": boxes, "masks": masks, "labels": torch.tensor([0, 0, 1]), "scores": torch.tensor([0.7, 0.7, 0.8])}
    target = {
        "boxes": boxes[[0, 2]],
        "masks": masks[[0, 2]],
        "labels": torch.tensor([0, 1]),
        "iscrowd": torch.tensor([0, 1]),
    }
    empty_pred = {key: value[:0] for key, value in pred.items()}
    empty_target = {key: value[:0] for key, value in target.items()}
    return [pred, empty_pred], [target, empty_target]


@pytest.mark.parametrize("iou_type", ["bbox", "segm", ("bbox", "segm")])
@pytest.mark.parametrize("average", ["macro", "micro"])
@pytest.mark.parametrize("max_dets", [[1, 10, 100], [1, 10, 500]])
def test_ultrafast_complete_outputs_and_lifecycle(iou_type, average, max_dets):
    """Preserve every tensor, including full arrays, ties, crowds and empty image state."""
    preds, targets = _inputs()
    kwargs = {
        "iou_type": iou_type,
        "average": average,
        "max_detection_thresholds": max_dets,
        "class_metrics": True,
        "extended_summary": True,
    }
    reference = MeanAveragePrecision(**kwargs)
    candidate = MeanAveragePrecision(backend="ultrafast", **kwargs)
    for metric in (reference, candidate):
        metric.update(deepcopy(preds[:1]), deepcopy(targets[:1]))
    torch.testing.assert_close(candidate.compute(), reference.compute(), rtol=0, atol=0, equal_nan=True)
    for metric in (reference, candidate):
        metric.update(deepcopy(preds[1:]), deepcopy(targets[1:]))
    expected = reference.compute()
    torch.testing.assert_close(candidate.compute(), expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        pickle.loads(pickle.dumps(candidate)).compute(), expected, rtol=0, atol=0, equal_nan=True
    )
    candidate.reset()
    candidate.update(deepcopy(preds), deepcopy(targets))
    torch.testing.assert_close(candidate.compute(), expected, rtol=0, atol=0, equal_nan=True)
    # Metric.compute squeezes scalar tensors; the functional API retains their shape.
    expected_functional = mean_average_precision(deepcopy(preds), deepcopy(targets), **kwargs)
    actual = mean_average_precision(deepcopy(preds), deepcopy(targets), backend="ultrafast", **kwargs)
    torch.testing.assert_close(actual, expected_functional, rtol=0, atol=0, equal_nan=True)


def test_ultrafast_without_reference_backends(monkeypatch, tmp_path):
    """The optional backend must not require either reference package to be available."""
    for module_name in ("torchmetrics.detection.mean_ap", "torchmetrics.detection.helpers"):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "_PYCOCOTOOLS_AVAILABLE", False)
        monkeypatch.setattr(module, "_FASTER_COCO_EVAL_AVAILABLE", False)
    monkeypatch.setattr("torchmetrics.detection.helpers._PYCOCOTOOLS_GREATER_EQUAL_2_0_9", False)
    backend = CocoBackend("ultrafast")
    assert backend.coco.__module__.startswith("ultrafast_pycocotools.")
    assert backend.cocoeval.__module__.startswith("ultrafast_pycocotools.")
    preds, targets = _inputs()
    metric = MeanAveragePrecision(backend="ultrafast")
    metric.update(preds, targets)
    assert metric.compute()["map"] == 1
    metric.tm_to_coco(str(tmp_path / "detections"))
    restored_preds, restored_targets = metric.coco_to_tm(
        str(tmp_path / "detections_preds.json"), str(tmp_path / "detections_target.json"), backend="ultrafast"
    )
    restored = MeanAveragePrecision(backend="ultrafast")
    restored.update(restored_preds, restored_targets)
    torch.testing.assert_close(restored.compute(), metric.compute(), rtol=0, atol=0)


def test_ultrafast_missing_or_old_dependency(monkeypatch):
    """Selecting an unavailable supported version gives installation guidance."""
    monkeypatch.setattr("torchmetrics.detection.helpers._ULTRAFAST_COCO_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match=r"ultrafast-pycocotools>=0\.1\.11"):
        _ = CocoBackend("ultrafast").coco
