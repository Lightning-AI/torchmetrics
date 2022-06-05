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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from tests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index
from torchmetrics.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from torchmetrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)

seed_all(42)


def _sk_confusion_matrix_binary(preds, target, normalize=None, ignore_index=None):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    return sk_confusion_matrix(y_true=target, y_pred=preds, normalize=normalize)


@pytest.mark.parametrize("input", _binary_cases)
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("ignore_index", [None, -1])
class TestBinaryConfusionMatrix(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_confusion_matrix(self, input, ddp, normalize, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryConfusionMatrix,
            sk_metric=partial(_sk_confusion_matrix_binary, normalize=normalize, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_confusion_matrix_functional(self, input, normalize, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_confusion_matrix,
            sk_metric=partial(_sk_confusion_matrix_binary, normalize=normalize, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )


def _sk_confusion_matrix_multiclass(preds, target, normalize=None, ignore_index=None):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()

    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    return sk_confusion_matrix(y_true=target, y_pred=preds, normalize=normalize)


@pytest.mark.parametrize("input", _multiclass_cases)
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("ignore_index", [None, -1])
class TestMulticlassConfusionMatrix(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_confusion_matrix(self, input, ddp, normalize, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassConfusionMatrix,
            sk_metric=partial(_sk_confusion_matrix_multiclass, normalize=normalize, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_confusion_matrix_functional(self, input, normalize, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_confusion_matrix,
            sk_metric=partial(_sk_confusion_matrix_multiclass, normalize=normalize, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )


def _sk_confusion_matrix_multilabel(preds, target, normalize=None, ignore_index=None):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    confmat = []
    for i in range(preds.shape[1]):
        p, t = preds[:, i], target[:, i]
        if ignore_index is not None:
            idx = t == ignore_index
            t = t[~idx]
            p = p[~idx]
        confmat.append(sk_confusion_matrix(t, p, normalize=normalize))
    return np.stack(confmat, axis=0)


@pytest.mark.parametrize("input", _multilabel_cases)
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("ignore_index", [None, -1])
class TestMultilabelConfusionMatrix(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_confusion_matrix(self, input, ddp, normalize, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelConfusionMatrix,
            sk_metric=partial(_sk_confusion_matrix_multilabel, normalize=normalize, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    def test_multilabel_confusion_matrix_functional(self, input, normalize, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_confusion_matrix,
            sk_metric=partial(_sk_confusion_matrix_multilabel, normalize=normalize, ignore_index=ignore_index),
            metric_args={
                "num_labels": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )


def test_warning_on_nan():
    preds = torch.randint(3, size=(20,))
    target = torch.randint(3, size=(20,))

    with pytest.warns(
        UserWarning,
        match=".* NaN values found in confusion matrix have been replaced with zeros.",
    ):
        multiclass_confusion_matrix(preds, target, num_classes=5, normalize="true")


# -------------------------- Old stuff --------------------------

# def _sk_cm_binary_prob(preds, target, normalize=None):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


# def _sk_cm_binary(preds, target, normalize=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


# def _sk_cm_multilabel_prob(preds, target, normalize=None):
#     sk_preds = (preds.numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.numpy()

#     cm = sk_multilabel_confusion_matrix(y_true=sk_target, y_pred=sk_preds)
#     if normalize is not None:
#         if normalize == "true":
#             cm = cm / cm.sum(axis=1, keepdims=True)
#         elif normalize == "pred":
#             cm = cm / cm.sum(axis=0, keepdims=True)
#         elif normalize == "all":
#             cm = cm / cm.sum()
#         cm[np.isnan(cm)] = 0
#     return cm


# def _sk_cm_multilabel(preds, target, normalize=None):
#     sk_preds = preds.numpy()
#     sk_target = target.numpy()

#     cm = sk_multilabel_confusion_matrix(y_true=sk_target, y_pred=sk_preds)
#     if normalize is not None:
#         if normalize == "true":
#             cm = cm / cm.sum(axis=1, keepdims=True)
#         elif normalize == "pred":
#             cm = cm / cm.sum(axis=0, keepdims=True)
#         elif normalize == "all":
#             cm = cm / cm.sum()
#         cm[np.isnan(cm)] = 0
#     return cm


# def _sk_cm_multiclass_prob(preds, target, normalize=None):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


# def _sk_cm_multiclass(preds, target, normalize=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


# def _sk_cm_multidim_multiclass_prob(preds, target, normalize=None):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


# def _sk_cm_multidim_multiclass(preds, target, normalize=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


# @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
# @pytest.mark.parametrize(
#     "preds, target, sk_metric, num_classes, multilabel",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_cm_binary_prob, 2, False),
#         (_input_binary_logits.preds, _input_binary_logits.target, _sk_cm_binary_prob, 2, False),
#         (_input_binary.preds, _input_binary.target, _sk_cm_binary, 2, False),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_cm_multilabel_prob, NUM_CLASSES, True),
#         (_input_mlb_logits.preds, _input_mlb_logits.target, _sk_cm_multilabel_prob, NUM_CLASSES, True),
#         (_input_mlb.preds, _input_mlb.target, _sk_cm_multilabel, NUM_CLASSES, True),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_cm_multiclass_prob, NUM_CLASSES, False),
#         (_input_mcls_logits.preds, _input_mcls_logits.target, _sk_cm_multiclass_prob, NUM_CLASSES, False),
#         (_input_mcls.preds, _input_mcls.target, _sk_cm_multiclass, NUM_CLASSES, False),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_cm_multidim_multiclass_prob, NUM_CLASSES, False),
#         (_input_mdmc.preds, _input_mdmc.target, _sk_cm_multidim_multiclass, NUM_CLASSES, False),
#     ],
# )
# class TestConfusionMatrix(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_confusion_matrix(
#         self, normalize, preds, target, sk_metric, num_classes, multilabel, ddp, dist_sync_on_step
#     ):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=ConfusionMatrix,
#             sk_metric=partial(sk_metric, normalize=normalize),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={
#                 "num_classes": num_classes,
#                 "threshold": THRESHOLD,
#                 "normalize": normalize,
#                 "multilabel": multilabel,
#             },
#         )

#     def test_confusion_matrix_functional(self, normalize, preds, target, sk_metric, num_classes, multilabel):
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=confusion_matrix,
#             sk_metric=partial(sk_metric, normalize=normalize),
#             metric_args={
#                 "num_classes": num_classes,
#                 "threshold": THRESHOLD,
#                 "normalize": normalize,
#                 "multilabel": multilabel,
#             },
#         )

#     def test_confusion_matrix_differentiability(self, normalize, preds, target, sk_metric, num_classes, multilabel):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=ConfusionMatrix,
#             metric_functional=confusion_matrix,
#             metric_args={
#                 "num_classes": num_classes,
#                 "threshold": THRESHOLD,
#                 "normalize": normalize,
#                 "multilabel": multilabel,
#             },
#         )


# def test_warning_on_nan(tmpdir):
#     preds = torch.randint(3, size=(20,))
#     target = torch.randint(3, size=(20,))

#     with pytest.warns(
#         UserWarning,
#         match=".* nan values found in confusion matrix have been replaced with zeros.",
#     ):
#         confusion_matrix(preds, target, num_classes=5, normalize="true")
