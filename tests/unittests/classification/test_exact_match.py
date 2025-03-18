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

import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid

from torchmetrics.classification.exact_match import ExactMatch, MulticlassExactMatch, MultilabelExactMatch
from torchmetrics.functional.classification.exact_match import multiclass_exact_match, multilabel_exact_match
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index
from unittests.classification._inputs import _multiclass_cases, _multilabel_cases

seed_all(42)


def _reference_exact_match_multiclass(preds, target, ignore_index, multidim_average):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)
    preds = preds.numpy()
    target = target.numpy()

    if ignore_index is not None:
        preds = np.copy(preds)
        preds[target == ignore_index] = ignore_index

    correct = (preds == target).sum(-1) == preds.shape[1]
    correct = correct.sum() if multidim_average == "global" else correct
    total = len(preds) if multidim_average == "global" else 1
    return correct / total


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassExactMatch(MetricTester):
    """Test class for `MulticlassExactMatch` metric."""

    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_exact_match(self, ddp, inputs, ignore_index, multidim_average):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if target.ndim < 3:
            pytest.skip("non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassExactMatch,
            reference_metric=partial(
                _reference_exact_match_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "num_classes": NUM_CLASSES,
                "multidim_average": multidim_average,
            },
        )

    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_exact_match_functional(self, inputs, ignore_index, multidim_average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if target.ndim < 3:
            pytest.skip("non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_exact_match,
            reference_metric=partial(
                _reference_exact_match_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "num_classes": NUM_CLASSES,
                "multidim_average": multidim_average,
            },
        )

    def test_multiclass_exact_match_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassExactMatch,
            metric_functional=multiclass_exact_match,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_exact_match_half_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassExactMatch,
            metric_functional=multiclass_exact_match,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_exact_match_half_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassExactMatch,
            metric_functional=multiclass_exact_match,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def _reference_exact_match_multilabel(preds, target, ignore_index, multidim_average):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)

    if ignore_index is not None:
        target = np.copy(target)
        target[target == ignore_index] = -1

    if multidim_average == "global":
        preds = np.moveaxis(preds, 1, -1).reshape(-1, NUM_CLASSES)
        target = np.moveaxis(target, 1, -1).reshape(-1, NUM_CLASSES)
        correct = ((preds == target).sum(1) == NUM_CLASSES).sum()
        total = preds.shape[0]
    else:
        correct = ((preds == target).sum(1) == NUM_CLASSES).sum(1)
        total = preds.shape[2]
    return correct / total


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelExactMatch(MetricTester):
    """Test class for `MultilabelExactMatch` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_multilabel_exact_match(self, ddp, inputs, ignore_index, multidim_average):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelExactMatch,
            reference_metric=partial(
                _reference_exact_match_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_multilabel_exact_match_functional(self, inputs, ignore_index, multidim_average):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_exact_match,
            reference_metric=partial(
                _reference_exact_match_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_multilabel_exact_match_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelExactMatch,
            metric_functional=multilabel_exact_match,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_exact_match_half_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelExactMatch,
            metric_functional=multilabel_exact_match,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_exact_match_half_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelExactMatch,
            metric_functional=multilabel_exact_match,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (MulticlassExactMatch, {"task": "multiclass", "num_classes": 3}),
        (MultilabelExactMatch, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=ExactMatch):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
