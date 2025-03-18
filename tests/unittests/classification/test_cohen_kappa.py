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
from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa

from torchmetrics.classification.cohen_kappa import BinaryCohenKappa, CohenKappa, MulticlassCohenKappa
from torchmetrics.functional.classification.cohen_kappa import binary_cohen_kappa, multiclass_cohen_kappa
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases

seed_all(42)


def _reference_sklearn_cohen_kappa_binary(preds, target, weights=None, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return sk_cohen_kappa(y1=target, y2=preds, weights=weights)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryCohenKappa(MetricTester):
    """Test class for `BinaryCohenKappa` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_cohen_kappa(self, inputs, ddp, weights, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryCohenKappa,
            reference_metric=partial(_reference_sklearn_cohen_kappa_binary, weights=weights, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_confusion_matrix_functional(self, inputs, weights, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_cohen_kappa,
            reference_metric=partial(_reference_sklearn_cohen_kappa_binary, weights=weights, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_cohen_kappa_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryCohenKappa,
            metric_functional=binary_cohen_kappa,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_cohen_kappa_dtypes_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryCohenKappa,
            metric_functional=binary_cohen_kappa,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_confusion_matrix_dtypes_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryCohenKappa,
            metric_functional=binary_cohen_kappa,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_cohen_kappa_multiclass(preds, target, weights, ignore_index=None):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    target, preds = remove_ignore_index(target=target, preds=preds, ignore_index=ignore_index)
    return sk_cohen_kappa(y1=target, y2=preds, weights=weights)


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassCohenKappa(MetricTester):
    """Test class for `MulticlassCohenKappa` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_multiclass_cohen_kappa(self, inputs, ddp, weights, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassCohenKappa,
            reference_metric=partial(
                _reference_sklearn_cohen_kappa_multiclass, weights=weights, ignore_index=ignore_index
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_confusion_matrix_functional(self, inputs, weights, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_cohen_kappa,
            reference_metric=partial(
                _reference_sklearn_cohen_kappa_multiclass, weights=weights, ignore_index=ignore_index
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_cohen_kappa_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassCohenKappa,
            metric_functional=multiclass_cohen_kappa,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_cohen_kappa_dtypes_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassCohenKappa,
            metric_functional=multiclass_cohen_kappa,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_confusion_matrix_dtypes_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassCohenKappa,
            metric_functional=multiclass_cohen_kappa,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryCohenKappa, {"task": "binary"}),
        (MulticlassCohenKappa, {"task": "multiclass", "num_classes": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=CohenKappa):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
