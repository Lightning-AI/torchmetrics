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
from functools import partial

import pytest
import torch
from torch import IntTensor, Tensor
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.detection.diou import DistanceIntersectionOverUnion
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.functional.detection.ciou import complete_intersection_over_union
from torchmetrics.functional.detection.diou import distance_intersection_over_union
from torchmetrics.functional.detection.giou import generalized_intersection_over_union
from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_13

# todo: check if some older versions have these functions too?
if _TORCHVISION_GREATER_EQUAL_0_13:
    from torchvision.ops import box_iou as tv_iou
    from torchvision.ops import complete_box_iou as tv_ciou
    from torchvision.ops import distance_box_iou as tv_diou
    from torchvision.ops import generalized_box_iou as tv_giou
else:
    tv_iou, tv_ciou, tv_diou, tv_giou = ..., ..., ..., ...

from unittests.helpers.testers import MetricTester


def _tv_wrapper(preds, target, base_fn, aggregate=True, iou_threshold=None):
    out = base_fn(preds, target)
    if iou_threshold is not None:
        out[out < iou_threshold] = 0
    if aggregate:
        return out.diag().mean()
    return out


def _tv_wrapper_class(preds, target, base_fn, respect_labels, iou_threshold, class_metrics):
    iou = []
    classes = []
    for p, t in zip(preds, target):
        out = base_fn(p["boxes"], t["boxes"])
        if iou_threshold is not None:
            out[out < iou_threshold] = -1
        if respect_labels:
            labels_eq = p["labels"].unsqueeze(1) == t["labels"].unsqueeze(0)
            out[~labels_eq] = -1
        iou.append(out)
        classes.append(t["labels"])
    score = torch.cat([i[i != -1] for i in iou]).mean()
    base_name = {tv_ciou: "ciou", tv_diou: "diou", tv_giou: "giou", tv_iou: "iou"}[base_fn]

    result = {f"{base_name}": score.cpu()}
    if class_metrics:
        for cl in torch.cat(classes).unique().tolist():
            class_score, numel = 0, 0
            for s, c in zip(iou, classes):
                masked_s = s[:, c == cl]
                class_score += masked_s[masked_s != -1].sum()
                numel += masked_s[masked_s != -1].numel()
            result.update({f"{base_name}/cl_{cl}": class_score.cpu() / numel})

    return result


_preds_fn = (
    torch.tensor([
        [296.55, 93.96, 314.97, 152.79],
        [328.94, 97.05, 342.49, 122.98],
        [356.62, 95.47, 372.33, 147.55],
    ])
    .unsqueeze(0)
    .repeat(4, 1, 1)
)
_target_fn = (
    torch.tensor([
        [300.00, 100.00, 315.00, 150.00],
        [330.00, 100.00, 350.00, 125.00],
        [350.00, 100.00, 375.00, 150.00],
    ])
    .unsqueeze(0)
    .repeat(4, 1, 1)
)

_preds_class = [
    [
        {
            "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
            "labels": torch.tensor([4, 5]),
        }
    ],
    [
        {
            "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
            "labels": torch.tensor([4, 5]),
        }
    ],
    [
        {
            "boxes": torch.tensor([[328.94, 97.05, 342.49, 122.98]]),
            "labels": torch.tensor([4]),
        },
        {
            "boxes": torch.tensor([[356.62, 95.47, 372.33, 147.55]]),
            "labels": torch.tensor([4]),
        },
    ],
    [
        {
            "boxes": torch.tensor([[328.94, 97.05, 342.49, 122.98]]),
            "labels": torch.tensor([5]),
        },
        {
            "boxes": torch.tensor([[356.62, 95.47, 372.33, 147.55]]),
            "labels": torch.tensor([5]),
        },
    ],
]
_target_class = [
    [
        {
            "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
            "labels": torch.tensor([5]),
        }
    ],
    [
        {
            "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
            "labels": torch.tensor([5]),
        }
    ],
    [
        {
            "boxes": torch.tensor([[330.00, 100.00, 350.00, 125.00]]),
            "labels": torch.tensor([4]),
        },
        {
            "boxes": torch.tensor([[350.00, 100.00, 375.00, 150.00]]),
            "labels": torch.tensor([4]),
        },
    ],
    [
        {
            "boxes": torch.tensor([[330.00, 100.00, 350.00, 125.00]]),
            "labels": torch.tensor([5]),
        },
        {
            "boxes": torch.tensor([[350.00, 100.00, 375.00, 150.00]]),
            "labels": torch.tensor([4]),
        },
    ],
]


def _add_noise(x, scale=10):
    """Add noise to boxes and labels to make testing non-deterministic."""
    if isinstance(x, torch.Tensor):
        return x + scale * torch.rand_like(x)
    for batch in x:
        for sample in batch:
            sample["boxes"] = _add_noise(sample["boxes"], scale)
            sample["labels"] += abs(torch.randint_like(sample["labels"], 0, 10))
    return x


@pytest.mark.parametrize(
    "class_metric, functional_metric, reference_metric",
    [
        (IntersectionOverUnion, intersection_over_union, tv_iou),
        (CompleteIntersectionOverUnion, complete_intersection_over_union, tv_ciou),
        (DistanceIntersectionOverUnion, distance_intersection_over_union, tv_diou),
        (GeneralizedIntersectionOverUnion, generalized_intersection_over_union, tv_giou),
    ],
)
@pytest.mark.skipif(not _TORCHVISION_GREATER_EQUAL_0_13, reason="test requires torchvision >= 0.13")
class TestIntersectionMetrics(MetricTester):
    """Tester class for the different intersection metrics."""

    @pytest.mark.parametrize(
        ("preds", "target"), [(_preds_class, _target_class), (_add_noise(_preds_class), _add_noise(_target_class))]
    )
    @pytest.mark.parametrize("respect_labels", [True, False])
    @pytest.mark.parametrize("iou_threshold", [None, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("class_metrics", [True, False])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_intersection_class(
        self,
        class_metric,
        functional_metric,
        reference_metric,
        preds,
        target,
        respect_labels,
        iou_threshold,
        class_metrics,
        ddp,
    ):
        """Test class implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=class_metric,
            reference_metric=partial(
                _tv_wrapper_class,
                base_fn=reference_metric,
                respect_labels=respect_labels,
                iou_threshold=iou_threshold,
                class_metrics=class_metrics,
            ),
            metric_args={
                "respect_labels": respect_labels,
                "iou_threshold": iou_threshold,
                "class_metrics": class_metrics,
            },
            check_batch=not class_metrics,
        )

    @pytest.mark.parametrize(
        ("preds", "target"),
        [
            (_preds_fn, _target_fn),
            (_add_noise(_preds_fn), _add_noise(_target_fn)),
        ],
    )
    @pytest.mark.parametrize("aggregate", [True, False])
    @pytest.mark.parametrize("iou_threshold", [None, 0.5, 0.7, 0.9])
    def test_intersection_function(
        self, class_metric, functional_metric, reference_metric, preds, target, aggregate, iou_threshold
    ):
        """Test functional implementation for correctness."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=partial(
                _tv_wrapper, base_fn=reference_metric, aggregate=aggregate, iou_threshold=iou_threshold
            ),
            metric_args={"aggregate": aggregate, "iou_threshold": iou_threshold},
        )

    def test_error_on_wrong_input(self, class_metric, functional_metric, reference_metric):
        """Test class input validation."""
        metric = class_metric()

        metric.update([], [])  # no error

        with pytest.raises(ValueError, match="Expected argument `preds` to be of type Sequence"):
            metric.update(Tensor(), [])  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `target` to be of type Sequence"):
            metric.update([], Tensor())  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have the same length"):
            metric.update([{}], [{}, {}])

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `boxes` key"):
            metric.update(
                [{"scores": Tensor(), "labels": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `labels` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `boxes` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor, "labels": IntTensor}],
                [{"labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `labels` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor, "labels": IntTensor}],
                [{"boxes": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": [], "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all labels in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": []}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `target` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": [], "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all labels in `target` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": []}],
            )


def test_corner_case():
    """See issue: https://github.com/Lightning-AI/torchmetrics/issues/1921."""
    preds = [
        {
            "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00], [298.55, 98.96, 314.97, 151.79]]),
            "scores": torch.tensor([0.236, 0.56]),
            "labels": torch.tensor([4, 5]),
        }
    ]

    target = [
        {
            "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00], [298.55, 98.96, 314.97, 151.79]]),
            "labels": torch.tensor([4, 5]),
        }
    ]

    metric = IntersectionOverUnion(class_metrics=True, iou_threshold=0.75, respect_labels=True)
    iou = metric(preds, target)
    for val in iou.values():
        assert val == torch.tensor(1.0)
