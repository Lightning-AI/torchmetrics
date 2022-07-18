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
from sklearn.metrics import matthews_corrcoef as sk_matthews_corrcoef

from torchmetrics.classification.matthews_corrcoef import (
    BinaryMatthewsCorrCoef,
    MulticlassMatthewsCorrCoef,
    MultilabelMatthewsCorrCoef,
)
from torchmetrics.functional.classification.matthews_corrcoef import (
    binary_matthews_corrcoef,
    multiclass_matthews_corrcoef,
    multilabel_matthews_corrcoef,
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_matthews_corrcoef_binary(preds, target, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target, preds)
    return sk_matthews_corrcoef(y_true=target, y_pred=preds)


@pytest.mark.parametrize("input", _binary_cases)
class TestBinaryMatthewsCorrCoef(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_matthews_corrcoef(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryMatthewsCorrCoef,
            sk_metric=partial(_sk_matthews_corrcoef_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_matthews_corrcoef_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_matthews_corrcoef,
            sk_metric=partial(_sk_matthews_corrcoef_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_matthews_corrcoef_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryMatthewsCorrCoef,
            metric_functional=binary_matthews_corrcoef,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_matthews_corrcoef_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryMatthewsCorrCoef,
            metric_functional=binary_matthews_corrcoef,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_matthews_corrcoef_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryMatthewsCorrCoef,
            metric_functional=binary_matthews_corrcoef,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _sk_matthews_corrcoef_multiclass(preds, target, ignore_index=None):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    target, preds = remove_ignore_index(target, preds)
    return sk_matthews_corrcoef(y_true=target, y_pred=preds)


@pytest.mark.parametrize("input", _multiclass_cases)
class TestMulticlassMatthewsCorrCoef(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_matthews_corrcoef(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassMatthewsCorrCoef,
            sk_metric=partial(_sk_matthews_corrcoef_multiclass, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_matthews_corrcoef_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_matthews_corrcoef,
            sk_metric=partial(_sk_matthews_corrcoef_multiclass, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_matthews_corrcoef_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassMatthewsCorrCoef,
            metric_functional=multiclass_matthews_corrcoef,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_matthews_corrcoef_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassMatthewsCorrCoef,
            metric_functional=multiclass_matthews_corrcoef,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_matthews_corrcoef_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassMatthewsCorrCoef,
            metric_functional=multiclass_matthews_corrcoef,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def _sk_matthews_corrcoef_multilabel(preds, target, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(target, preds)
    return sk_matthews_corrcoef(y_true=target, y_pred=preds)


@pytest.mark.parametrize("input", _multilabel_cases)
class TestMultilabelMatthewsCorrCoef(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_matthews_corrcoef(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelMatthewsCorrCoef,
            sk_metric=partial(_sk_matthews_corrcoef_multilabel, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_matthews_corrcoef_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_matthews_corrcoef,
            sk_metric=partial(_sk_matthews_corrcoef_multilabel, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multilabel_matthews_corrcoef_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelMatthewsCorrCoef,
            metric_functional=multilabel_matthews_corrcoef,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_matthews_corrcoef_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelMatthewsCorrCoef,
            metric_functional=multilabel_matthews_corrcoef,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_matthews_corrcoef_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelMatthewsCorrCoef,
            metric_functional=multilabel_matthews_corrcoef,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_zero_case_in_multiclass():
    """Cases where the denominator in the matthews corrcoef is 0, the score should return 0."""
    # Example where neither 1 or 2 is present in the target tensor
    out = multiclass_matthews_corrcoef(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]), 3)
    assert out == 0.0


# -------------------------- Old stuff --------------------------

# def _sk_matthews_corrcoef_binary_prob(preds, target):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_binary(preds, target):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_multilabel_prob(preds, target):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_multilabel(preds, target):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_multiclass_prob(preds, target):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_multiclass(preds, target):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_multidim_multiclass_prob(preds, target):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# def _sk_matthews_corrcoef_multidim_multiclass(preds, target):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


# @pytest.mark.parametrize(
#     "preds, target, sk_metric, num_classes",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_matthews_corrcoef_binary_prob, 2),
#         (_input_binary.preds, _input_binary.target, _sk_matthews_corrcoef_binary, 2),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_matthews_corrcoef_multilabel_prob, 2),
#         (_input_mlb.preds, _input_mlb.target, _sk_matthews_corrcoef_multilabel, 2),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_matthews_corrcoef_multiclass_prob, NUM_CLASSES),
#         (_input_mcls.preds, _input_mcls.target, _sk_matthews_corrcoef_multiclass, NUM_CLASSES),
#         (
#           _input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_matthews_corrcoef_multidim_multiclass_prob, NUM_CLASSES
#         ),
#         (_input_mdmc.preds, _input_mdmc.target, _sk_matthews_corrcoef_multidim_multiclass, NUM_CLASSES),
#     ],
# )
# class TestMatthewsCorrCoef(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_matthews_corrcoef(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=MatthewsCorrCoef,
#             sk_metric=sk_metric,
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={
#                 "num_classes": num_classes,
#                 "threshold": THRESHOLD,
#             },
#         )

#     def test_matthews_corrcoef_functional(self, preds, target, sk_metric, num_classes):
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=matthews_corrcoef,
#             sk_metric=sk_metric,
#             metric_args={
#                 "num_classes": num_classes,
#                 "threshold": THRESHOLD,
#             },
#         )

#     def test_matthews_corrcoef_differentiability(self, preds, target, sk_metric, num_classes):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=MatthewsCorrCoef,
#             metric_functional=matthews_corrcoef,
#             metric_args={
#                 "num_classes": num_classes,
#                 "threshold": THRESHOLD,
#             },
#         )


# def test_zero_case():
#     """Cases where the denominator in the matthews corrcoef is 0, the score should return 0."""
#     # Example where neither 1 or 2 is present in the target tensor
#     out = matthews_corrcoef(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]), 3)
#     assert out == 0.0
