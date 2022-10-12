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

import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa

from torchmetrics.classification.cohen_kappa import BinaryCohenKappa, MulticlassCohenKappa
from torchmetrics.functional.classification.cohen_kappa import binary_cohen_kappa, multiclass_cohen_kappa
from unittests.classification.inputs import _binary_cases, _multiclass_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_cohen_kappa_binary(preds, target, weights=None, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_cohen_kappa(y1=target, y2=preds, weights=weights)


@pytest.mark.parametrize("input", _binary_cases)
class TestBinaryCohenKappa(MetricTester):
    atol = 1e-5

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_cohen_kappa(self, input, ddp, weights, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryCohenKappa,
            sk_metric=partial(_sk_cohen_kappa_binary, weights=weights, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_confusion_matrix_functional(self, input, weights, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_cohen_kappa,
            sk_metric=partial(_sk_cohen_kappa_binary, weights=weights, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_cohen_kappa_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryCohenKappa,
            metric_functional=binary_cohen_kappa,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_cohen_kappa_dtypes_cpu(self, input, dtype):
        preds, target = input

        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
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
    def test_binary_confusion_matrix_dtypes_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryCohenKappa,
            metric_functional=binary_cohen_kappa,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _sk_cohen_kappa_multiclass(preds, target, weights, ignore_index=None):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_cohen_kappa(y1=target, y2=preds, weights=weights)


@pytest.mark.parametrize("input", _multiclass_cases)
class TestMulticlassCohenKappa(MetricTester):
    atol = 1e-5

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_cohen_kappa(self, input, ddp, weights, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassCohenKappa,
            sk_metric=partial(_sk_cohen_kappa_multiclass, weights=weights, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_confusion_matrix_functional(self, input, weights, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_cohen_kappa,
            sk_metric=partial(_sk_cohen_kappa_multiclass, weights=weights, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "weights": weights,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_cohen_kappa_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassCohenKappa,
            metric_functional=multiclass_cohen_kappa,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_cohen_kappa_dtypes_cpu(self, input, dtype):
        preds, target = input

        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
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
    def test_multiclass_confusion_matrix_dtypes_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassCohenKappa,
            metric_functional=multiclass_cohen_kappa,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )
