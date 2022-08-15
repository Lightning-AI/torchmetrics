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
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from torchmetrics.classification.auroc import BinaryAUROC, MulticlassAUROC, MultilabelAUROC
from torchmetrics.functional.classification.auroc import binary_auroc, multiclass_auroc, multilabel_auroc
from torchmetrics.functional.classification.roc import binary_roc
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6, _TORCH_LOWER_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_auroc_binary(preds, target, max_fpr=None, ignore_index=None):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_roc_auc_score(target, preds, max_fpr=max_fpr)


@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryAUROC(MetricTester):
    @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_auroc(self, input, ddp, max_fpr, ignore_index):
        if max_fpr is not None and _TORCH_LOWER_1_6:
            pytest.skip("requires torch v1.6 or higher to test max_fpr argument")
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryAUROC,
            sk_metric=partial(_sk_auroc_binary, max_fpr=max_fpr, ignore_index=ignore_index),
            metric_args={
                "max_fpr": max_fpr,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_auroc_functional(self, input, max_fpr, ignore_index):
        if max_fpr is not None and _TORCH_LOWER_1_6:
            pytest.skip("requires torch v1.6 or higher to test max_fpr argument")
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_auroc,
            sk_metric=partial(_sk_auroc_binary, max_fpr=max_fpr, ignore_index=ignore_index),
            metric_args={
                "max_fpr": max_fpr,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_auroc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_auroc_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_auroc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize("threshold_fn", [lambda x: x, lambda x: x.numpy().tolist()], ids=["as tensor", "as list"])
    def test_binary_auroc_threshold_arg(self, input, threshold_fn):
        preds, target = input

        for pred, true in zip(preds, target):
            _, _, t = binary_roc(pred, true, thresholds=None)
            ap1 = binary_auroc(pred, true, thresholds=None)
            ap2 = binary_auroc(pred, true, thresholds=threshold_fn(t.flip(0)))
            assert torch.allclose(ap1, ap2)


def _sk_auroc_multiclass(preds, target, average="macro", ignore_index=None):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    return sk_roc_auc_score(target, preds, average=average, multi_class="ovr", labels=list(range(NUM_CLASSES)))


@pytest.mark.parametrize(
    "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassAUROC(MetricTester):
    @pytest.mark.parametrize("average", ["macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_auroc(self, input, average, ddp, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassAUROC,
            sk_metric=partial(_sk_auroc_multiclass, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_auroc_functional(self, input, average, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_auroc,
            sk_metric=partial(_sk_auroc_multiclass, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_auroc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_auroc_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_auroc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["macro", "weighted", None])
    def test_multiclass_auroc_threshold_arg(self, input, average):
        preds, target = input
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 2)) + 1e-6  # rounding will simulate binning
            ap1 = multiclass_auroc(pred, true, num_classes=NUM_CLASSES, average=average, thresholds=None)
            ap2 = multiclass_auroc(
                pred, true, num_classes=NUM_CLASSES, average=average, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2)


def _sk_auroc_multilabel(preds, target, average="macro", ignore_index=None):
    if ignore_index is None:
        if preds.ndim > 2:
            target = target.transpose(2, 1).reshape(-1, NUM_CLASSES)
            preds = preds.transpose(2, 1).reshape(-1, NUM_CLASSES)
        target = target.numpy()
        preds = preds.numpy()
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        return sk_roc_auc_score(target, preds, average=average, max_fpr=None)
    if average == "micro":
        return _sk_auroc_binary(preds.flatten(), target.flatten(), max_fpr=None, ignore_index=ignore_index)
    res = []
    for i in range(NUM_CLASSES):
        res.append(_sk_auroc_binary(preds[:, i], target[:, i], max_fpr=None, ignore_index=ignore_index))
    if average == "macro":
        return np.array(res)[~np.isnan(res)].mean()
    if average == "weighted":
        weights = ((target == 1).sum([0, 2]) if target.ndim == 3 else (target == 1).sum(0)).numpy()
        weights = weights / sum(weights)
        return (np.array(res) * weights)[~np.isnan(res)].sum()
    return res


@pytest.mark.parametrize(
    "input", (_multilabel_cases[1], _multilabel_cases[2], _multilabel_cases[4], _multilabel_cases[5])
)
class TestMultilabelAUROC(MetricTester):
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multilabel_auroc(self, input, ddp, average, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelAUROC,
            sk_metric=partial(_sk_auroc_multilabel, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multilabel_auroc_functional(self, input, average, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_auroc,
            sk_metric=partial(_sk_auroc_multilabel, average=average, ignore_index=ignore_index),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_auroc_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_auroc_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if dtype == torch.half and not ((0 < preds) & (preds < 1)).all():
            pytest.xfail(reason="half support for torch.softmax on cpu not implemented")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_auroc_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_auroc_threshold_arg(self, input, average):
        preds, target = input
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = torch.tensor(np.round(pred.numpy(), 1)) + 1e-6  # rounding will simulate binning
            ap1 = multilabel_auroc(pred, true, num_labels=NUM_CLASSES, average=average, thresholds=None)
            ap2 = multilabel_auroc(
                pred, true, num_labels=NUM_CLASSES, average=average, thresholds=torch.linspace(0, 1, 100)
            )
            assert torch.allclose(ap1, ap2)


# -------------------------- Old stuff --------------------------


# def _sk_auroc_binary_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
#     # todo: `multi_class` is unused
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()
#     return sk_roc_auc_score(y_true=sk_target, y_score=sk_preds, average=average, max_fpr=max_fpr)


# def _sk_auroc_multiclass_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
#     sk_preds = preds.reshape(-1, num_classes).numpy()
#     sk_target = target.view(-1).numpy()
#     return sk_roc_auc_score(
#         y_true=sk_target,
#         y_score=sk_preds,
#         average=average,
#         max_fpr=max_fpr,
#         multi_class=multi_class,
#     )


# def _sk_auroc_multidim_multiclass_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
#     sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
#     sk_target = target.view(-1).numpy()
#     return sk_roc_auc_score(
#         y_true=sk_target,
#         y_score=sk_preds,
#         average=average,
#         max_fpr=max_fpr,
#         multi_class=multi_class,
#     )


# def _sk_auroc_multilabel_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
#     sk_preds = preds.reshape(-1, num_classes).numpy()
#     sk_target = target.reshape(-1, num_classes).numpy()
#     return sk_roc_auc_score(
#         y_true=sk_target,
#         y_score=sk_preds,
#         average=average,
#         max_fpr=max_fpr,
#         multi_class=multi_class,
#     )


# def _sk_auroc_multilabel_multidim_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
#     sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
#     sk_target = target.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
#     return sk_roc_auc_score(
#         y_true=sk_target,
#         y_score=sk_preds,
#         average=average,
#         max_fpr=max_fpr,
#         multi_class=multi_class,
#     )


# @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
# @pytest.mark.parametrize(
#     "preds, target, sk_metric, num_classes",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_auroc_binary_prob, 1),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_auroc_multiclass_prob, NUM_CLASSES),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_auroc_multidim_multiclass_prob, NUM_CLASSES),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_auroc_multilabel_prob, NUM_CLASSES),
#         (_input_mlmd_prob.preds, _input_mlmd_prob.target, _sk_auroc_multilabel_multidim_prob, NUM_CLASSES),
#     ],
# )
# class TestAUROC(MetricTester):
#     @pytest.mark.parametrize("average", ["macro", "weighted", "micro"])
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_auroc(self, preds, target, sk_metric, num_classes, average, max_fpr, ddp, dist_sync_on_step):
#         # max_fpr different from None is not support in multi class
#         if max_fpr is not None and num_classes != 1:
#             pytest.skip("max_fpr parameter not support for multi class or multi label")

#         # max_fpr only supported for torch v1.6 or higher
#         if max_fpr is not None and _TORCH_LOWER_1_6:
#             pytest.skip("requires torch v1.6 or higher to test max_fpr argument")

#         # average='micro' only supported for multilabel
#         if average == "micro" and preds.ndim > 2 and preds.ndim == target.ndim + 1:
#             pytest.skip("micro argument only support for multilabel input")

#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=AUROC,
#             sk_metric=partial(sk_metric, num_classes=num_classes, average=average, max_fpr=max_fpr),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"num_classes": num_classes, "average": average, "max_fpr": max_fpr},
#         )

#     @pytest.mark.parametrize("average", ["macro", "weighted", "micro"])
#     def test_auroc_functional(self, preds, target, sk_metric, num_classes, average, max_fpr):
#         # max_fpr different from None is not support in multi class
#         if max_fpr is not None and num_classes != 1:
#             pytest.skip("max_fpr parameter not support for multi class or multi label")

#         # max_fpr only supported for torch v1.6 or higher
#         if max_fpr is not None and _TORCH_LOWER_1_6:
#             pytest.skip("requires torch v1.6 or higher to test max_fpr argument")

#         # average='micro' only supported for multilabel
#         if average == "micro" and preds.ndim > 2 and preds.ndim == target.ndim + 1:
#             pytest.skip("micro argument only support for multilabel input")

#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=auroc,
#             sk_metric=partial(sk_metric, num_classes=num_classes, average=average, max_fpr=max_fpr),
#             metric_args={"num_classes": num_classes, "average": average, "max_fpr": max_fpr},
#         )

#     def test_auroc_differentiability(self, preds, target, sk_metric, num_classes, max_fpr):
#         # max_fpr different from None is not support in multi class
#         if max_fpr is not None and num_classes != 1:
#             pytest.skip("max_fpr parameter not support for multi class or multi label")

#         # max_fpr only supported for torch v1.6 or higher
#         if max_fpr is not None and _TORCH_LOWER_1_6:
#             pytest.skip("requires torch v1.6 or higher to test max_fpr argument")

#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=AUROC,
#             metric_functional=auroc,
#             metric_args={"num_classes": num_classes, "max_fpr": max_fpr},
#         )


# def test_error_on_different_mode():
#     """test that an error is raised if the user pass in data of different modes (binary, multi-label, multi-
#     class)"""
#     metric = AUROC()
#     # pass in multi-class data
#     metric.update(torch.randn(10, 5).softmax(dim=-1), torch.randint(0, 5, (10,)))
#     with pytest.raises(ValueError, match=r"The mode of data.* should be constant.*"):
#         # pass in multi-label data
#         metric.update(torch.rand(10, 5), torch.randint(0, 2, (10, 5)))


# def test_error_multiclass_no_num_classes():
#     with pytest.raises(
#         ValueError, match="Detected input to `multiclass` but you did not provide `num_classes` argument"
#     ):
#         _ = auroc(torch.randn(20, 3).softmax(dim=-1), torch.randint(3, (20,)))


# @pytest.mark.parametrize("device", ["cpu", "cuda"])
# def test_weighted_with_empty_classes(device):
#     """Tests that weighted multiclass AUROC calculation yields the same results if a new but empty class exists.

#     Tests that the proper warnings and errors are raised
#     """
#     if not torch.cuda.is_available() and device == "cuda":
#         pytest.skip("Test requires gpu to run")

#     preds = torch.tensor(
#         [
#             [0.90, 0.05, 0.05],
#             [0.05, 0.90, 0.05],
#             [0.05, 0.05, 0.90],
#             [0.85, 0.05, 0.10],
#             [0.10, 0.10, 0.80],
#         ]
#     ).to(device)
#     target = torch.tensor([0, 1, 1, 2, 2]).to(device)
#     num_classes = 3
#     _auroc = auroc(preds, target, average="weighted", num_classes=num_classes)

#     # Add in a class with zero observations at second to last index
#     preds = torch.cat(
#         (preds[:, : num_classes - 1], torch.rand_like(preds[:, 0:1]), preds[:, num_classes - 1 :]), axis=1
#     )
#     # Last class (2) gets moved to 3
#     target[target == num_classes - 1] = num_classes
#     with pytest.warns(UserWarning, match="Class 2 had 0 observations, omitted from AUROC calculation"):
#         _auroc_empty_class = auroc(preds, target, average="weighted", num_classes=num_classes + 1)
#     assert _auroc == _auroc_empty_class

#     target = torch.zeros_like(target)
#     with pytest.raises(ValueError, match="Found 1 non-empty class in `multiclass` AUROC calculation"):
#         _ = auroc(preds, target, average="weighted", num_classes=num_classes + 1)


# def test_warnings_on_missing_class():
#     """Test that a warning is given if either the positive or negative class is missing."""
#     metric = AUROC()
#     # no positive samples
#     warning = (
#         "No positive samples in targets, true positive value should be meaningless."
#         " Returning zero tensor in true positive score"
#     )
#     with pytest.warns(UserWarning, match=warning):
#         score = metric(torch.randn(10).sigmoid(), torch.zeros(10).int())
#     assert score == 0

#     warning = (
#         "No negative samples in targets, false positive value should be meaningless."
#         " Returning zero tensor in false positive score"
#     )
#     with pytest.warns(UserWarning, match=warning):
#         score = metric(torch.randn(10).sigmoid(), torch.ones(10).int())
#     assert score == 0
