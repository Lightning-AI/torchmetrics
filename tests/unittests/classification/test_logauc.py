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
from scipy.special import expit as sigmoid
from scipy.special import softmax
from tdc.evaluator import range_logAUC
from torchmetrics.functional.classification.logauc import binary_logauc, multiclass_logauc, multilabel_logauc
from torchmetrics.classification.logauc import BinaryLogAUC
from unittests import NUM_CLASSES
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _binary_compare_implementation(preds, target, fpr_range):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    return range_logAUC(target, preds, FPR_range=fpr_range)


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryAUROC(MetricTester):
    """Test class for `BinaryAUROC` metric."""

    @pytest.mark.parametrize("fpr_range", [(0.001, 0.1), (0.01, 0.1), (0.1, 0.2)])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_auroc(self, inputs, ddp, fpr_range):
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
    def test_binary_auroc_functional(self, inputs, fpr_range):
        """Test functional implementation of metric."""
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_logauc,
            reference_metric=partial(_binary_compare_implementation, fpr_range=fpr_range),
            metric_args={
                "fpr_range": fpr_range,
                "thresholds": None,
            },
        )

    # def test_binary_auroc_differentiability(self, inputs):
    #     """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
    #     preds, target = inputs
    #     self.run_differentiability_test(
    #         preds=preds,
    #         target=target,
    #         metric_module=BinaryAUROC,
    #         metric_functional=binary_auroc,
    #         metric_args={"thresholds": None},
    #     )

    # @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    # def test_binary_auroc_dtype_cpu(self, inputs, dtype):
    #     """Test dtype support of the metric on CPU."""
    #     preds, target = inputs

    #     if (preds < 0).any() and dtype == torch.half:
    #         pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
    #     self.run_precision_test_cpu(
    #         preds=preds,
    #         target=target,
    #         metric_module=BinaryAUROC,
    #         metric_functional=binary_auroc,
    #         metric_args={"thresholds": None},
    #         dtype=dtype,
    #     )

    # @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    # @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    # def test_binary_auroc_dtype_gpu(self, inputs, dtype):
    #     """Test dtype support of the metric on GPU."""
    #     preds, target = inputs
    #     self.run_precision_test_gpu(
    #         preds=preds,
    #         target=target,
    #         metric_module=BinaryAUROC,
    #         metric_functional=binary_auroc,
    #         metric_args={"thresholds": None},
    #         dtype=dtype,
    #     )

    # @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    # def test_binary_auroc_threshold_arg(self, inputs, threshold_fn):
    #     """Test that different types of `thresholds` argument lead to same result."""
    #     preds, target = inputs

    #     for pred, true in zip(preds, target):
    #         _, _, t = binary_roc(pred, true, thresholds=None)
    #         ap1 = binary_auroc(pred, true, thresholds=None)
    #         ap2 = binary_auroc(pred, true, thresholds=threshold_fn(t.flip(0)))
    #         assert torch.allclose(ap1, ap2)
