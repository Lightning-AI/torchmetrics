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
from scipy.special import softmax
from tdc.evaluator import range_logAUC
from torchmetrics.classification.logauc import BinaryLogAUC, MulticlassLogAUC
from torchmetrics.functional.classification.logauc import binary_logauc, multiclass_logauc
from torchmetrics.functional.classification.roc import binary_roc

from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

seed_all(42)


def _binary_compare_implementation(preds, target, fpr_range, ignore_index=None):
    """Binary comparison function for logauc."""
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return range_logAUC(target, preds, FPR_range=fpr_range)


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryLogAUC(MetricTester):
    """Test class for `BinaryLogAUC` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    def test_binary_logauc(self, inputs, ddp, fpr_range):
        """Test class implementation of metric."""
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryLogAUC,
            reference_metric=partial(_binary_compare_implementation, fpr_range=fpr_range),
            metric_args={
                "fpr_range": fpr_range,
                "thresholds": None,
            },
        )

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_logauc_functional(self, inputs, fpr_range, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_logauc,
            reference_metric=partial(_binary_compare_implementation, fpr_range=fpr_range, ignore_index=ignore_index),
            metric_args={
                "fpr_range": fpr_range,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_logauc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryLogAUC,
            metric_functional=binary_logauc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_logauc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryLogAUC,
            metric_functional=binary_logauc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_logauc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryLogAUC,
            metric_functional=binary_logauc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_logauc_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs

        for pred, true in zip(preds, target):
            _, _, t = binary_roc(pred, true, thresholds=None)
            ap1 = binary_logauc(pred, true, thresholds=None)
            ap2 = binary_logauc(pred, true, thresholds=threshold_fn(t.flip(0)))
            assert torch.allclose(ap1, ap2)


# def _multiclass_compare_implementation(preds, target, fpr_range, average):
#     """Multiclass comparison function for logauc."""
#     preds = preds.permute(0, 2, 1).reshape(-1, NUM_CLASSES).numpy() if preds.ndim == 3 else preds.numpy()
#     target = target.flatten().numpy()
#     if not ((preds > 0) & (preds < 1)).all():
#         preds = softmax(preds, 1)

#     scores = []
#     for i in range(NUM_CLASSES):
#         p, t = preds[:, i], (target == i).astype(int)
#         scores.append(range_logAUC(t, p, FPR_range=fpr_range))
#     if average == "macro":
#         return np.mean(scores)
#     return scores


# @pytest.mark.parametrize(
#     "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
# )
# class TestMulticlassLogAUC(MetricTester):
#     """Test class for `MulticlassLogAUC` metric."""

#     @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
#     @pytest.mark.parametrize("average", ["macro", "weighted"])
#     @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
#     def test_multiclass_logauc(self, inputs, fpr_range, average, ddp):
#         """Test class implementation of metric."""
#         preds, target = inputs
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=MulticlassLogAUC,
#             reference_metric=partial(_multiclass_compare_implementation, fpr_range=fpr_range, average=average),
#             metric_args={
#                 "thresholds": None,
#                 "num_classes": NUM_CLASSES,
#                 "fpr_range": fpr_range,
#                 "average": average,
#             },
#         )

#     @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
#     @pytest.mark.parametrize("average", ["macro", None])
#     def test_multiclass_logauc_functional(self, inputs, fpr_range, average):
#         """Test functional implementation of metric."""
#         preds, target = inputs
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=multiclass_logauc,
#             reference_metric=partial(_multiclass_compare_implementation, fpr_range=fpr_range, average=average),
#             metric_args={
#                 "thresholds": None,
#                 "num_classes": NUM_CLASSES,
#                 "fpr_range": fpr_range,
#                 "average": average,
#             },
#         )

# def test_multiclass_logauc_differentiability(self, inputs):
#     """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
#     preds, target = inputs
#     self.run_differentiability_test(
#         preds=preds,
#         target=target,
#         metric_module=MulticlassLogAUC,
#         metric_functional=multiclass_logauc,
#         metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
#     )

# @pytest.mark.parametrize("dtype", [torch.half, torch.double])
# def test_multiclass_logauc_dtype_cpu(self, inputs, dtype):
#     """Test dtype support of the metric on CPU."""
#     preds, target = inputs

#     if dtype == torch.half and not ((preds > 0) & (preds < 1)).all():
#         pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
#     self.run_precision_test_cpu(
#         preds=preds,
#         target=target,
#         metric_module=MulticlassLogAUC,
#         metric_functional=multiclass_logauc,
#         metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
#         dtype=dtype,
#     )

# @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
# @pytest.mark.parametrize("dtype", [torch.half, torch.double])
# def test_multiclass_logauc_dtype_gpu(self, inputs, dtype):
#     """Test dtype support of the metric on GPU."""
#     preds, target = inputs
#     self.run_precision_test_gpu(
#         preds=preds,
#         target=target,
#         metric_module=MulticlassLogAUC,
#         metric_functional=multiclass_logauc,
#         metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
#         dtype=dtype,
#     )

# @pytest.mark.parametrize("average", ["macro", "weighted", None])
# def test_multiclass_logauc_threshold_arg(self, inputs, average):
#     """Test that different types of `thresholds` argument lead to same result."""
#     preds, target = inputs
#     if (preds < 0).any():
#         preds = preds.softmax(dim=-1)
#     for pred, true in zip(preds, target):
#         pred = torch.tensor(np.round(pred.numpy(), 2)) + 1e-6  # rounding will simulate binning
#         ap1 = multiclass_logauc(pred, true, num_classes=NUM_CLASSES, average=average, thresholds=None)
#         ap2 = multiclass_logauc(
#             pred, true, num_classes=NUM_CLASSES, average=average, thresholds=torch.linspace(0, 1, 100)
#         )
#         assert torch.allclose(ap1, ap2)
