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
from scipy.special import softmax
from sklearn.metrics import hinge_loss as sk_hinge
from sklearn.preprocessing import OneHotEncoder

from torchmetrics.classification.hinge import BinaryHingeLoss, MulticlassHingeLoss
from torchmetrics.functional.classification.hinge import binary_hinge_loss, multiclass_hinge_loss
from unittests.classification.inputs import _binary_cases, _multiclass_cases
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index, remove_ignore_index

torch.manual_seed(42)


def _sk_binary_hinge_loss(preds, target, ignore_index):
    preds = preds.numpy().flatten()
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)

    target, preds = remove_ignore_index(target, preds, ignore_index)
    target = 2 * target - 1
    return sk_hinge(target, preds)


@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryHingeLoss(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_hinge_loss(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryHingeLoss,
            sk_metric=partial(_sk_binary_hinge_loss, ignore_index=ignore_index),
            metric_args={
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_hinge_loss_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_hinge_loss,
            sk_metric=partial(_sk_binary_hinge_loss, ignore_index=ignore_index),
            metric_args={
                "ignore_index": ignore_index,
            },
        )

    def test_binary_hinge_loss_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryHingeLoss,
            metric_functional=binary_hinge_loss,
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_hinge_loss_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half:
            pytest.xfail(reason="torch.clamp does not support cpu + half")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryHingeLoss,
            metric_functional=binary_hinge_loss,
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_hinge_loss_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryHingeLoss,
            metric_functional=binary_hinge_loss,
            dtype=dtype,
        )


def _sk_multiclass_hinge_loss(preds, target, multiclass_mode, ignore_index):
    preds = preds.numpy()
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = softmax(preds, 1)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target, preds = remove_ignore_index(target, preds, ignore_index)

    if multiclass_mode == "one-vs-all":
        enc = OneHotEncoder()
        enc.fit(target.reshape(-1, 1))
        target = enc.transform(target.reshape(-1, 1)).toarray()
        target = 2 * target - 1
        result = np.zeros(preds.shape[1])
        for i in range(result.shape[0]):
            result[i] = sk_hinge(y_true=target[:, i], pred_decision=preds[:, i])
        return result
    else:
        return sk_hinge(target, preds)


@pytest.mark.parametrize(
    "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassHingeLoss(MetricTester):
    @pytest.mark.parametrize("multiclass_mode", ["crammer-singer", "one-vs-all"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_hinge_loss(self, input, ddp, multiclass_mode, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassHingeLoss,
            sk_metric=partial(_sk_multiclass_hinge_loss, multiclass_mode=multiclass_mode, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "multiclass_mode": multiclass_mode,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("multiclass_mode", ["crammer-singer", "one-vs-all"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_hinge_loss_functional(self, input, multiclass_mode, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_hinge_loss,
            sk_metric=partial(_sk_multiclass_hinge_loss, multiclass_mode=multiclass_mode, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "multiclass_mode": multiclass_mode,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_hinge_loss_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassHingeLoss,
            metric_functional=multiclass_hinge_loss,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_hinge_loss_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half:
            pytest.xfail(reason="torch.clamp does not support cpu + half")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassHingeLoss,
            metric_functional=multiclass_hinge_loss,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_hinge_loss_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassHingeLoss,
            metric_functional=multiclass_hinge_loss,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )
