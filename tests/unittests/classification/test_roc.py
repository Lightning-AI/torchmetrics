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
from sklearn.metrics import roc_curve as sk_roc_curve

from torchmetrics.classification.roc import BinaryROC, MulticlassROC, MultilabelROC
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _nan_to_zero(args):
    for a in args:
        a[np.isnan(a)] = 0
    return args


def _sk_roc_binary(preds, target, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    fpr, tpr, thresholds = sk_roc_curve(target, preds, drop_intermediate=False)
    thresholds[0] = 1.0
    return _nan_to_zero([fpr, tpr, thresholds])


@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryROC(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_roc(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryROC,
            sk_metric=partial(_sk_roc_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_roc_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_roc,
            sk_metric=partial(_sk_roc_binary, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_roc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryROC,
            metric_functional=binary_roc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_roc_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryROC,
            metric_functional=binary_roc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_roc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryROC,
            metric_functional=binary_roc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_roc_threshold_arg(self, input, threshold_fn):
        preds, target = input

        for pred, true in zip(preds, target):
            p1, r1, t1 = binary_roc(pred, true, thresholds=None)
            p2, r2, t2 = binary_roc(pred, true, thresholds=threshold_fn(t1))

            assert torch.allclose(p1, p2)
            assert torch.allclose(r1, r2)
            assert torch.allclose(t1, t2)


def _sk_roc_multiclass(preds, target, ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target, preds, ignore_index)

    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        target_temp = np.zeros_like(target)
        target_temp[target == i] = 1
        res = sk_roc_curve(target_temp, preds[:, i], drop_intermediate=False)
        res[2][0] = 1.0

        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return _nan_to_zero(fpr), _nan_to_zero(tpr), _nan_to_zero(thresholds)


@pytest.mark.parametrize(
    "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassROC(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_roc(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassROC,
            sk_metric=partial(_sk_roc_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_roc_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_roc,
            sk_metric=partial(_sk_roc_multiclass, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_roc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassROC,
            metric_functional=multiclass_roc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_roc_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassROC,
            metric_functional=multiclass_roc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_roc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassROC,
            metric_functional=multiclass_roc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multiclass_roc_threshold_arg(self, input, threshold_fn):
        preds, target = input
        for pred, true in zip(preds, target):
            p1, r1, t1 = multiclass_roc(pred, true, num_classes=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multiclass_roc(pred, true, num_classes=NUM_CLASSES, thresholds=threshold_fn(t))

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)


def _sk_roc_multilabel(preds, target, ignore_index=None):
    fpr, tpr, thresholds = [], [], []
    for i in range(NUM_CLASSES):
        res = _sk_roc_binary(preds[:, i], target[:, i], ignore_index)
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return fpr, tpr, thresholds


@pytest.mark.parametrize(
    "input", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelROC(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_roc(self, input, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelROC,
            sk_metric=partial(_sk_roc_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_roc_functional(self, input, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_roc,
            sk_metric=partial(_sk_roc_multilabel, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_roc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelROC,
            metric_functional=multilabel_roc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_roc_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelROC,
            metric_functional=multilabel_roc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_roc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelROC,
            metric_functional=multilabel_roc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_multilabel_roc_threshold_arg(self, input, threshold_fn):
        preds, target = input
        for pred, true in zip(preds, target):
            p1, r1, t1 = multilabel_roc(pred, true, num_labels=NUM_CLASSES, thresholds=None)
            for i, t in enumerate(t1):
                p2, r2, t2 = multilabel_roc(pred, true, num_labels=NUM_CLASSES, thresholds=threshold_fn(t))

                assert torch.allclose(p1[i], p2[i])
                assert torch.allclose(r1[i], r2[i])
                assert torch.allclose(t1[i], t2)


# -------------------------- Old stuff --------------------------

# def _sk_roc_curve(y_true, probas_pred, num_classes: int = 1, multilabel: bool = False):
#     """Adjusted comparison function that can also handles multiclass."""
#     if num_classes == 1:
#         return sk_roc_curve(y_true, probas_pred, drop_intermediate=False)

#     fpr, tpr, thresholds = [], [], []
#     for i in range(num_classes):
#         if multilabel:
#             y_true_temp = y_true[:, i]
#         else:
#             y_true_temp = np.zeros_like(y_true)
#             y_true_temp[y_true == i] = 1

#         res = sk_roc_curve(y_true_temp, probas_pred[:, i], drop_intermediate=False)
#         fpr.append(res[0])
#         tpr.append(res[1])
#         thresholds.append(res[2])
#     return fpr, tpr, thresholds


# def _sk_roc_binary_prob(preds, target, num_classes=1):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


# def _sk_roc_multiclass_prob(preds, target, num_classes=1):
#     sk_preds = preds.reshape(-1, num_classes).numpy()
#     sk_target = target.view(-1).numpy()

#     return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


# def _sk_roc_multidim_multiclass_prob(preds, target, num_classes=1):
#     sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
#     sk_target = target.view(-1).numpy()
#     return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


# def _sk_roc_multilabel_prob(preds, target, num_classes=1):
#     sk_preds = preds.numpy()
#     sk_target = target.numpy()
#     return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, multilabel=True)


# def _sk_roc_multilabel_multidim_prob(preds, target, num_classes=1):
#     sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
#     sk_target = target.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
#     return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, multilabel=True)


# @pytest.mark.parametrize(
#     "preds, target, sk_metric, num_classes",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_roc_binary_prob, 1),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_roc_multiclass_prob, NUM_CLASSES),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_roc_multidim_multiclass_prob, NUM_CLASSES),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_roc_multilabel_prob, NUM_CLASSES),
#         (_input_mlmd_prob.preds, _input_mlmd_prob.target, _sk_roc_multilabel_multidim_prob, NUM_CLASSES),
#     ],
# )
# class TestROC(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_roc(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=ROC,
#             sk_metric=partial(sk_metric, num_classes=num_classes),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"num_classes": num_classes},
#         )

#     def test_roc_functional(self, preds, target, sk_metric, num_classes):
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=roc,
#             sk_metric=partial(sk_metric, num_classes=num_classes),
#             metric_args={"num_classes": num_classes},
#         )

#     def test_roc_differentiability(self, preds, target, sk_metric, num_classes):
#         self.run_differentiability_test(
#             preds,
#             target,
#             metric_module=ROC,
#             metric_functional=roc,
#             metric_args={"num_classes": num_classes},
#         )


# @pytest.mark.parametrize(
#     ["pred", "target", "expected_tpr", "expected_fpr"],
#     [
#         ([0, 1], [0, 1], [0, 1, 1], [0, 0, 1]),
#         ([1, 0], [0, 1], [0, 0, 1], [0, 1, 1]),
#         ([1, 1], [1, 0], [0, 1], [0, 1]),
#         ([1, 0], [1, 0], [0, 1, 1], [0, 0, 1]),
#         ([0.5, 0.5], [0, 1], [0, 1], [0, 1]),
#     ],
# )
# def test_roc_curve(pred, target, expected_tpr, expected_fpr):
#     fpr, tpr, thresh = roc(tensor(pred), tensor(target))

#     assert fpr.shape == tpr.shape
#     assert fpr.size(0) == thresh.size(0)
#     assert torch.allclose(fpr, tensor(expected_fpr).to(fpr))
#     assert torch.allclose(tpr, tensor(expected_tpr).to(tpr))


# def test_warnings_on_missing_class():
#     """Test that a warning is given if either the positive or negative class is missing."""
#     metric = ROC()
#     # no positive samples
#     warning = (
#         "No positive samples in targets, true positive value should be meaningless."
#         " Returning zero tensor in true positive score"
#     )
#     with pytest.warns(UserWarning, match=warning):
#         _, tpr, _ = metric(torch.randn(10).sigmoid(), torch.zeros(10))
#     assert all(tpr == 0)

#     warning = (
#         "No negative samples in targets, false positive value should be meaningless."
#         " Returning zero tensor in false positive score"
#     )
#     with pytest.warns(UserWarning, match=warning):
#         fpr, _, _ = metric(torch.randn(10).sigmoid(), torch.ones(10))
#     assert all(fpr == 0)
