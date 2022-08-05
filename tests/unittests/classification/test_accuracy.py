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
from sklearn.metrics import accuracy_score as sk_accuracy
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from torchmetrics.classification.accuracy import BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.functional.classification.accuracy import binary_accuracy, multiclass_accuracy, multilabel_accuracy
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_accuracy(target, preds):
    score = sk_accuracy(target, preds)
    return score if not np.isnan(score) else 0.0


def _sk_accuracy_binary(preds, target, ignore_index, multidim_average):
    if multidim_average == "global":
        preds = preds.view(-1).numpy()
        target = target.view(-1).numpy()
    else:
        preds = preds.numpy()
        target = target.numpy()

    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)

    if multidim_average == "global":
        target, preds = remove_ignore_index(target, preds, ignore_index)
        return _sk_accuracy(target, preds)
    else:
        res = []
        for pred, true in zip(preds, target):
            pred = pred.flatten()
            true = true.flatten()
            true, pred = remove_ignore_index(true, pred, ignore_index)
            res.append(_sk_accuracy(true, pred))
        return np.stack(res)


@pytest.mark.parametrize("input", _binary_cases)
class TestBinaryAccuracy(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ddp", [False, True])
    def test_binary_accuracy(self, ddp, input, ignore_index, multidim_average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryAccuracy,
            sk_metric=partial(_sk_accuracy_binary, ignore_index=ignore_index, multidim_average=multidim_average),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "multidim_average": multidim_average},
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_accuracy_functional(self, input, ignore_index, multidim_average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_accuracy,
            sk_metric=partial(_sk_accuracy_binary, ignore_index=ignore_index, multidim_average=multidim_average),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_accuracy_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryAccuracy,
            metric_functional=binary_accuracy,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_accuracy_half_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryAccuracy,
            metric_functional=binary_accuracy,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_accuracy_half_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryAccuracy,
            metric_functional=binary_accuracy,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _sk_accuracy_multiclass(preds, target, ignore_index, multidim_average, average):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)
    if multidim_average == "global":
        preds = preds.numpy().flatten()
        target = target.numpy().flatten()
        target, preds = remove_ignore_index(target, preds, ignore_index)
        if average == "micro":
            return _sk_accuracy(target, preds)
        confmat = sk_confusion_matrix(target, preds, labels=list(range(NUM_CLASSES)))
        acc_per_class = confmat.diagonal() / confmat.sum(axis=1)
        acc_per_class[np.isnan(acc_per_class)] = 0.0
        if average == "macro":
            return acc_per_class.mean()
        elif average == "weighted":
            weights = confmat.sum(1)
            return ((weights * acc_per_class) / weights.sum()).sum()
        else:
            return acc_per_class
    else:
        preds = preds.numpy()
        target = target.numpy()
        res = []
        for pred, true in zip(preds, target):
            pred = pred.flatten()
            true = true.flatten()
            true, pred = remove_ignore_index(true, pred, ignore_index)
            if average == "micro":
                res.append(_sk_accuracy(true, pred))
            else:
                confmat = sk_confusion_matrix(true, pred, labels=list(range(NUM_CLASSES)))
                acc_per_class = confmat.diagonal() / confmat.sum(axis=1)
                acc_per_class[np.isnan(acc_per_class)] = 0.0
                if average == "macro":
                    res.append(acc_per_class.mean())
                elif average == "weighted":
                    weights = confmat.sum(1)
                    score = ((weights * acc_per_class) / weights.sum()).sum()
                    res.append(0.0 if np.isnan(score) else score)
                else:
                    res.append(acc_per_class)
        return np.stack(res, 0)


@pytest.mark.parametrize("input", _multiclass_cases)
class TestMulticlassAccuracy(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_accuracy(self, ddp, input, ignore_index, multidim_average, average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassAccuracy,
            sk_metric=partial(
                _sk_accuracy_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multiclass_accuracy_functional(self, input, ignore_index, multidim_average, average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_accuracy,
            sk_metric=partial(
                _sk_accuracy_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    def test_multiclass_accuracy_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_accuracy_half_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_accuracy_half_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


_mc_k_target = torch.tensor([0, 1, 2])
_mc_k_preds = torch.tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])


@pytest.mark.parametrize(
    "k, preds, target, average, expected",
    [
        (1, _mc_k_preds, _mc_k_target, "micro", torch.tensor(2 / 3)),
        (2, _mc_k_preds, _mc_k_target, "micro", torch.tensor(3 / 3)),
    ],
)
def test_top_k(k, preds, target, average, expected):
    """A simple test to check that top_k works as expected."""
    class_metric = MulticlassAccuracy(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)
    assert torch.isclose(class_metric.compute(), expected)
    assert torch.isclose(multiclass_accuracy(preds, target, top_k=k, average=average, num_classes=3), expected)


def _sk_accuracy_multilabel(preds, target, ignore_index, multidim_average, average):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)

    if multidim_average == "global":
        if average == "micro":
            preds = preds.flatten()
            target = target.flatten()
            target, preds = remove_ignore_index(target, preds, ignore_index)
            return _sk_accuracy(target, preds)

        accuracy, weights = [], []
        for i in range(preds.shape[1]):
            pred, true = preds[:, i].flatten(), target[:, i].flatten()
            true, pred = remove_ignore_index(true, pred, ignore_index)
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            accuracy.append(_sk_accuracy(true, pred))
            weights.append(confmat[1, 1] + confmat[1, 0])
        res = np.stack(accuracy, axis=0)

        if average == "macro":
            return res.mean(0)
        elif average == "weighted":
            weights = np.stack(weights, 0).astype(float)
            weights_norm = weights.sum(-1, keepdims=True)
            weights_norm[weights_norm == 0] = 1.0
            return ((weights * res) / weights_norm).sum(-1)
        elif average is None or average == "none":
            return res
    else:
        accuracy, weights = [], []
        for i in range(preds.shape[0]):
            if average == "micro":
                pred, true = preds[i].flatten(), target[i].flatten()
                true, pred = remove_ignore_index(true, pred, ignore_index)
                accuracy.append(_sk_accuracy(true, pred))
                confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                weights.append(confmat[1, 1] + confmat[1, 0])
            else:
                scores, w = [], []
                for j in range(preds.shape[1]):
                    pred, true = preds[i, j], target[i, j]
                    true, pred = remove_ignore_index(true, pred, ignore_index)
                    scores.append(_sk_accuracy(true, pred))
                    confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                    w.append(confmat[1, 1] + confmat[1, 0])
                accuracy.append(np.stack(scores))
                weights.append(np.stack(w))
        if average == "micro":
            return np.array(accuracy)
        res = np.stack(accuracy, 0)
        if average == "macro":
            return res.mean(-1)
        elif average == "weighted":
            weights = np.stack(weights, 0).astype(float)
            weights_norm = weights.sum(-1, keepdims=True)
            weights_norm[weights_norm == 0] = 1.0
            return ((weights * res) / weights_norm).sum(-1)
        elif average is None or average == "none":
            return res


@pytest.mark.parametrize("input", _multilabel_cases)
class TestMultilabelAccuracy(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_accuracy(self, ddp, input, ignore_index, multidim_average, average):
        preds, target = input
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
            metric_class=MultilabelAccuracy,
            sk_metric=partial(
                _sk_accuracy_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_accuracy_functional(self, input, ignore_index, multidim_average, average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_accuracy,
            sk_metric=partial(
                _sk_accuracy_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    def test_multilabel_accuracy_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelAccuracy,
            metric_functional=multilabel_accuracy,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_accuracy_half_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAccuracy,
            metric_functional=multilabel_accuracy,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_accuracy_half_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAccuracy,
            metric_functional=multilabel_accuracy,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


# -------------------------- Old stuff --------------------------

# def _sk_accuracy(preds, target, subset_accuracy):
#     sk_preds, sk_target, mode = _input_format_classification(preds, target, threshold=THRESHOLD)
#     sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

#     if mode == DataType.MULTIDIM_MULTICLASS and not subset_accuracy:
#         sk_preds, sk_target = np.transpose(sk_preds, (0, 2, 1)), np.transpose(sk_target, (0, 2, 1))
#         sk_preds, sk_target = sk_preds.reshape(-1, sk_preds.shape[2]), sk_target.reshape(-1, sk_target.shape[2])
#     elif mode == DataType.MULTIDIM_MULTICLASS and subset_accuracy:
#         return np.all(sk_preds == sk_target, axis=(1, 2)).mean()
#     elif mode == DataType.MULTILABEL and not subset_accuracy:
#         sk_preds, sk_target = sk_preds.reshape(-1), sk_target.reshape(-1)

#     return sk_accuracy(y_true=sk_target, y_pred=sk_preds)


# @pytest.mark.parametrize(
#     "preds, target, subset_accuracy, mdmc_average",
#     [
#         (_input_binary_logits.preds, _input_binary_logits.target, False, None),
#         (_input_binary_prob.preds, _input_binary_prob.target, False, None),
#         (_input_binary.preds, _input_binary.target, False, None),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, True, None),
#         (_input_mlb_logits.preds, _input_mlb_logits.target, False, None),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, False, None),
#         (_input_mlb.preds, _input_mlb.target, True, None),
#         (_input_mlb.preds, _input_mlb.target, False, "global"),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, False, None),
#         (_input_mcls_logits.preds, _input_mcls_logits.target, False, None),
#         (_input_mcls.preds, _input_mcls.target, False, None),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, False, "global"),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, True, None),
#         (_input_mdmc.preds, _input_mdmc.target, False, "global"),
#         (_input_mdmc.preds, _input_mdmc.target, True, None),
#         (_input_mlmd_prob.preds, _input_mlmd_prob.target, True, None),
#         (_input_mlmd_prob.preds, _input_mlmd_prob.target, False, None),
#         (_input_mlmd.preds, _input_mlmd.target, True, None),
#         (_input_mlmd.preds, _input_mlmd.target, False, "global"),
#     ],
# )
# class TestAccuracies(MetricTester):
#     @pytest.mark.parametrize("ddp", [False, True])
#     @pytest.mark.parametrize("dist_sync_on_step", [False, True])
#     def test_accuracy_class(self, ddp, dist_sync_on_step, preds, target, subset_accuracy, mdmc_average):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=Accuracy,
#             sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy, "mdmc_average": mdmc_average},
#         )

#     def test_accuracy_fn(self, preds, target, subset_accuracy, mdmc_average):
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=accuracy,
#             sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
#             metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy},
#         )

#     def test_accuracy_differentiability(self, preds, target, subset_accuracy, mdmc_average):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=Accuracy,
#             metric_functional=accuracy,
#             metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy, "mdmc_average": mdmc_average},
#         )


# _l1to4 = [0.1, 0.2, 0.3, 0.4]
# _l1to4t3 = np.array([_l1to4, _l1to4, _l1to4])
# _l1to4t3_mcls = [_l1to4t3.T, _l1to4t3.T, _l1to4t3.T]

# # The preds in these examples always put highest probability on class 3, second highest on class 2,
# # third highest on class 1, and lowest on class 0
# _topk_preds_mcls = tensor([_l1to4t3, _l1to4t3]).float()
# _topk_target_mcls = tensor([[1, 2, 3], [2, 1, 0]])

# # This is like for MC case, but one sample in each batch is sabotaged with 0 class prediction :)
# _topk_preds_mdmc = tensor([_l1to4t3_mcls, _l1to4t3_mcls]).float()
# _topk_target_mdmc = tensor([[[1, 1, 0], [2, 2, 2], [3, 3, 3]], [[2, 2, 0], [1, 1, 1], [0, 0, 0]]])

# # Multilabel
# _ml_t1 = [0.8, 0.2, 0.8, 0.2]
# _ml_t2 = [_ml_t1, _ml_t1]
# _ml_ta2 = [[1, 0, 1, 1], [0, 1, 1, 0]]
# _av_preds_ml = tensor([_ml_t2, _ml_t2]).float()
# _av_target_ml = tensor([_ml_ta2, _ml_ta2])

# # Inputs with negative target values to be ignored
# Input = namedtuple("Input", ["preds", "target", "ignore_index", "result"])
# _binary_with_neg_tgt = Input(
#     preds=torch.tensor([0, 1, 0]), target=torch.tensor([0, 1, -1]), ignore_index=-1, result=torch.tensor(1.0)
# )
# _multiclass_logits_with_neg_tgt = Input(
#     preds=torch.tensor([[0.8, 0.1], [0.2, 0.7], [0.5, 0.5]]),
#     target=torch.tensor([0, 1, -1]),
#     ignore_index=-1,
#     result=torch.tensor(1.0),
# )
# _multidim_multiclass_with_neg_tgt = Input(
#     preds=torch.tensor([[0, 0], [1, 1], [0, 0]]),
#     target=torch.tensor([[0, 0], [-1, 1], [1, -1]]),
#     ignore_index=-1,
#     result=torch.tensor(0.75),
# )
# _multidim_multiclass_logits_with_neg_tgt = Input(
#     preds=torch.tensor([[[0.8, 0.7], [0.2, 0.4]], [[0.1, 0.2], [0.9, 0.8]], [[0.7, 0.9], [0.2, 0.4]]]),
#     target=torch.tensor([[0, 0], [-1, 1], [1, -1]]),
#     ignore_index=-1,
#     result=torch.tensor(0.75),
# )


# # Replace with a proper sk_metric test once sklearn 0.24 hits :)
# @pytest.mark.parametrize(
#     "preds, target, exp_result, k, subset_accuracy",
#     [
#         (_topk_preds_mcls, _topk_target_mcls, 1 / 6, 1, False),
#         (_topk_preds_mcls, _topk_target_mcls, 3 / 6, 2, False),
#         (_topk_preds_mcls, _topk_target_mcls, 5 / 6, 3, False),
#         (_topk_preds_mcls, _topk_target_mcls, 1 / 6, 1, True),
#         (_topk_preds_mcls, _topk_target_mcls, 3 / 6, 2, True),
#         (_topk_preds_mcls, _topk_target_mcls, 5 / 6, 3, True),
#         (_topk_preds_mdmc, _topk_target_mdmc, 1 / 6, 1, False),
#         (_topk_preds_mdmc, _topk_target_mdmc, 8 / 18, 2, False),
#         (_topk_preds_mdmc, _topk_target_mdmc, 13 / 18, 3, False),
#         (_topk_preds_mdmc, _topk_target_mdmc, 1 / 6, 1, True),
#         (_topk_preds_mdmc, _topk_target_mdmc, 2 / 6, 2, True),
#         (_topk_preds_mdmc, _topk_target_mdmc, 3 / 6, 3, True),
#         (_av_preds_ml, _av_target_ml, 5 / 8, None, False),
#         (_av_preds_ml, _av_target_ml, 0, None, True),
#     ],
# )
# def test_topk_accuracy(preds, target, exp_result, k, subset_accuracy):
#     topk = Accuracy(top_k=k, subset_accuracy=subset_accuracy, mdmc_average="global")

#     for batch in range(preds.shape[0]):
#         topk(preds[batch], target[batch])

#     assert topk.compute() == exp_result

#     # Test functional
#     total_samples = target.shape[0] * target.shape[1]

#     preds = preds.view(total_samples, 4, -1)
#     target = target.view(total_samples, -1)

#     assert accuracy(preds, target, top_k=k, subset_accuracy=subset_accuracy) == exp_result


# # Only MC and MDMC with probs input type should be accepted for top_k
# @pytest.mark.parametrize(
#     "preds, target",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target),
#         (_input_binary.preds, _input_binary.target),
#         (_input_mlb_prob.preds, _input_mlb_prob.target),
#         (_input_mlb.preds, _input_mlb.target),
#         (_input_mcls.preds, _input_mcls.target),
#         (_input_mdmc.preds, _input_mdmc.target),
#         (_input_mlmd_prob.preds, _input_mlmd_prob.target),
#         (_input_mlmd.preds, _input_mlmd.target),
#     ],
# )
# def test_topk_accuracy_wrong_input_types(preds, target):
#     topk = Accuracy(top_k=1)

#     with pytest.raises(ValueError):
#         topk(preds[0], target[0])

#     with pytest.raises(ValueError):
#         accuracy(preds[0], target[0], top_k=1)


# @pytest.mark.parametrize(
#     "average, mdmc_average, num_classes, inputs, ignore_index, top_k, threshold",
#     [
#         ("unknown", None, None, _input_binary, None, None, 0.5),
#         ("micro", "unknown", None, _input_binary, None, None, 0.5),
#         ("macro", None, None, _input_binary, None, None, 0.5),
#         ("micro", None, None, _input_mdmc_prob, None, None, 0.5),
#         ("micro", None, None, _input_binary_prob, 0, None, 0.5),
#         ("micro", None, None, _input_mcls_prob, NUM_CLASSES, None, 0.5),
#         ("micro", None, NUM_CLASSES, _input_mcls_prob, NUM_CLASSES, None, 0.5),
#         (None, None, None, _input_mcls_prob, None, 0, 0.5),
#         (None, None, None, _input_mcls_prob, None, None, 1.5),
#     ],
# )
# def test_wrong_params(average, mdmc_average, num_classes, inputs, ignore_index, top_k, threshold):
#     preds, target = inputs.preds, inputs.target

#     with pytest.raises(ValueError):
#         acc = Accuracy(
#             average=average,
#             mdmc_average=mdmc_average,
#             num_classes=num_classes,
#             ignore_index=ignore_index,
#             threshold=threshold,
#             top_k=top_k,
#         )
#         acc(preds[0], target[0])
#         acc.compute()

#     with pytest.raises(ValueError):
#         accuracy(
#             preds[0],
#             target[0],
#             average=average,
#             mdmc_average=mdmc_average,
#             num_classes=num_classes,
#             ignore_index=ignore_index,
#             threshold=threshold,
#             top_k=top_k,
#         )


# @pytest.mark.parametrize(
#     "preds_mc, target_mc, preds_ml, target_ml",
#     [
#         (
#             tensor([0, 1, 1, 1]),
#             tensor([2, 2, 1, 1]),
#             tensor([[0.8, 0.2, 0.8, 0.7], [0.6, 0.4, 0.6, 0.5]]),
#             tensor([[1, 0, 1, 1], [0, 0, 1, 0]]),
#         )
#     ],
# )
# def test_different_modes(preds_mc, target_mc, preds_ml, target_ml):
#     acc = Accuracy()
#     acc(preds_mc, target_mc)
#     with pytest.raises(ValueError, match="^[You cannot use]"):
#         acc(preds_ml, target_ml)


# _bin_t1 = [0.7, 0.6, 0.2, 0.1]
# _av_preds_bin = tensor([_bin_t1, _bin_t1]).float()
# _av_target_bin = tensor([[1, 0, 0, 0], [0, 1, 1, 0]])


# @pytest.mark.parametrize(
#     "preds, target, num_classes, exp_result, average, mdmc_average",
#     [
#         (_topk_preds_mcls, _topk_target_mcls, 4, 1 / 4, "macro", None),
#         (_topk_preds_mcls, _topk_target_mcls, 4, 1 / 6, "weighted", None),
#         (_topk_preds_mcls, _topk_target_mcls, 4, [0.0, 0.0, 0.0, 1.0], "none", None),
#         (_topk_preds_mcls, _topk_target_mcls, 4, 1 / 6, "samples", None),
#         (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 24, "macro", "samplewise"),
#         (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 6, "weighted", "samplewise"),
#         (_topk_preds_mdmc, _topk_target_mdmc, 4, [0.0, 0.0, 0.0, 1 / 6], "none", "samplewise"),
#         (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 6, "samples", "samplewise"),
#         (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 6, "samples", "global"),
#         (_av_preds_ml, _av_target_ml, 4, 5 / 8, "macro", None),
#         (_av_preds_ml, _av_target_ml, 4, 0.70000005, "weighted", None),
#         (_av_preds_ml, _av_target_ml, 4, [1 / 2, 1 / 2, 1.0, 1 / 2], "none", None),
#         (_av_preds_ml, _av_target_ml, 4, 5 / 8, "samples", None),
#     ],
# )
# def test_average_accuracy(preds, target, num_classes, exp_result, average, mdmc_average):
#     acc = Accuracy(num_classes=num_classes, average=average, mdmc_average=mdmc_average)

#     for batch in range(preds.shape[0]):
#         acc(preds[batch], target[batch])

#     assert (acc.compute() == tensor(exp_result)).all()

#     # Test functional
#     total_samples = target.shape[0] * target.shape[1]

#     preds = preds.view(total_samples, num_classes, -1)
#     target = target.view(total_samples, -1)

#     acc_score = accuracy(preds, target, num_classes=num_classes, average=average, mdmc_average=mdmc_average)
#     assert (acc_score == tensor(exp_result)).all()


# @pytest.mark.parametrize(
#     "preds, target, num_classes, exp_result, average, multiclass",
#     [
#         (_av_preds_bin, _av_target_bin, 2, 19 / 30, "macro", True),
#         (_av_preds_bin, _av_target_bin, 2, 5 / 8, "weighted", True),
#         (_av_preds_bin, _av_target_bin, 2, [3 / 5, 2 / 3], "none", True),
#         (_av_preds_bin, _av_target_bin, 2, 5 / 8, "samples", True),
#     ],
# )
# def test_average_accuracy_bin(preds, target, num_classes, exp_result, average, multiclass):
#     acc = Accuracy(num_classes=num_classes, average=average, multiclass=multiclass)

#     for batch in range(preds.shape[0]):
#         acc(preds[batch], target[batch])

#     assert (acc.compute() == tensor(exp_result)).all()

#     # Test functional
#     total_samples = target.shape[0] * target.shape[1]

#     preds = preds.view(total_samples, -1)
#     target = target.view(total_samples, -1)
#     acc_score = accuracy(preds, target, num_classes=num_classes, average=average, multiclass=multiclass)
#     assert (acc_score == tensor(exp_result)).all()


# @pytest.mark.parametrize("metric_class, metric_fn", [(Accuracy, accuracy)])
# @pytest.mark.parametrize(
#     "ignore_index, expected", [(None, torch.tensor([1.0, np.nan])), (0, torch.tensor([np.nan, np.nan]))]
# )
# def test_class_not_present(metric_class, metric_fn, ignore_index, expected):
#     """This tests that when metric is computed per class and a given class is not present in both the `preds` and
#     `target`, the resulting score is `nan`."""
#     preds = torch.tensor([0, 0, 0])
#     target = torch.tensor([0, 0, 0])
#     num_classes = 2

#     # test functional
#     result_fn = metric_fn(
#       preds, target, average=AverageMethod.NONE, num_classes=num_classes, ignore_index=ignore_index
#     )
#     assert torch.allclose(expected, result_fn, equal_nan=True)

#     # test class
#     cl_metric = metric_class(average=AverageMethod.NONE, num_classes=num_classes, ignore_index=ignore_index)
#     cl_metric(preds, target)
#     result_cl = cl_metric.compute()
#     assert torch.allclose(expected, result_cl, equal_nan=True)


# @pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
# def test_same_input(average):
#     preds = _input_miss_class.preds
#     target = _input_miss_class.target
#     preds_flat = torch.cat(list(preds), dim=0)
#     target_flat = torch.cat(list(target), dim=0)

#     mc = Accuracy(num_classes=NUM_CLASSES, average=average)
#     for i in range(NUM_BATCHES):
#         mc.update(preds[i], target[i])
#     class_res = mc.compute()
#     func_res = accuracy(preds_flat, target_flat, num_classes=NUM_CLASSES, average=average)
#     sk_res = sk_accuracy(target_flat, preds_flat)

#     assert torch.allclose(class_res, torch.tensor(sk_res).float())
#     assert torch.allclose(func_res, torch.tensor(sk_res).float())


# @pytest.mark.parametrize(
#     "preds, target, ignore_index, result",
#     [
#         (
#             _binary_with_neg_tgt.preds,
#             _binary_with_neg_tgt.target,
#             _binary_with_neg_tgt.ignore_index,
#             _binary_with_neg_tgt.result,
#         ),
#         (
#             _multiclass_logits_with_neg_tgt.preds,
#             _multiclass_logits_with_neg_tgt.target,
#             _multiclass_logits_with_neg_tgt.ignore_index,
#             _multiclass_logits_with_neg_tgt.result,
#         ),
#         (
#             _multidim_multiclass_with_neg_tgt.preds,
#             _multidim_multiclass_with_neg_tgt.target,
#             _multidim_multiclass_with_neg_tgt.ignore_index,
#             _multidim_multiclass_with_neg_tgt.result,
#         ),
#         (
#             _multidim_multiclass_logits_with_neg_tgt.preds,
#             _multidim_multiclass_logits_with_neg_tgt.target,
#             _multidim_multiclass_logits_with_neg_tgt.ignore_index,
#             _multidim_multiclass_logits_with_neg_tgt.result,
#         ),
#     ],
# )
# def test_negative_ignore_index(preds, target, ignore_index, result):
#     # We deduct -1 for an ignored index
#     num_classes = len(target.unique()) - 1

#     # Test class
#     acc = Accuracy(num_classes=num_classes, ignore_index=ignore_index)
#     acc_score = acc(preds, target)
#     assert torch.allclose(acc_score, result)
#     # Test functional metrics
#     acc_score = accuracy(preds, target, num_classes=num_classes, ignore_index=ignore_index)
#     assert torch.allclose(acc_score, result)

#     # If the ignore index is not set properly, we expect to see an error
#     ignore_index = None
#     # Test class
#     acc = Accuracy(num_classes=num_classes, ignore_index=ignore_index)
#     with pytest.raises(ValueError, match="^[The `target` has to be a non-negative tensor.]"):
#         acc_score = acc(preds, target)

#     # Test functional
#     with pytest.raises(ValueError, match="^[The `target` has to be a non-negative tensor.]"):
#         acc_score = accuracy(preds, target, num_classes=num_classes, ignore_index=ignore_index)


# def test_negmetric_noneavg(noneavg=_negmetric_noneavg):
#     acc = MetricWrapper(Accuracy(average="none", num_classes=noneavg["pred1"].shape[1]))
#     result1 = acc(noneavg["pred1"], noneavg["target1"])
#     assert torch.allclose(noneavg["res1"], result1, equal_nan=True)
#     result2 = acc(noneavg["pred2"], noneavg["target2"])
#     assert torch.allclose(noneavg["res2"], result2, equal_nan=True)
