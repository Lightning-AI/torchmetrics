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
from scipy.special import expit as sigmoid
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from tests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.functional.classification.stat_scores import (
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
)

seed_all(42)


def _sk_stat_scores_binary(preds, target, ignore_index, multidim_average):
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
        if ignore_index is not None:
            idx = target == ignore_index
            target = target[~idx]
            preds = preds[~idx]
        tn, fp, fn, tp = sk_confusion_matrix(y_true=target, y_pred=preds, labels=[0, 1]).ravel()
        return np.array([tp, fp, tn, fn, tp + fp + tn + fn])
    else:
        res = []
        for pred, true in zip(preds, target):
            if ignore_index is not None:
                idx = true == ignore_index
                true = true[~idx]
                pred = pred[~idx]
            tn, fp, fn, tp = sk_confusion_matrix(y_true=true, y_pred=pred, labels=[0, 1]).ravel()
            res.append(np.array([tp, fp, tn, fn, tp + fp + tn + fn]))
        return np.stack(res)


@pytest.mark.parametrize("input", _binary_cases)
@pytest.mark.parametrize("ignore_index", [None, 0, -1])
@pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
class TestBinaryStatScores(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_stat_scores(self, ddp, input, ignore_index, multidim_average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryStatScores,
            sk_metric=partial(_sk_stat_scores_binary, ignore_index=ignore_index, multidim_average=multidim_average),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "multidim_average": multidim_average},
        )

    def test_binary_stat_scores_functional(self, input, ignore_index, multidim_average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_stat_scores,
            sk_metric=partial(_sk_stat_scores_binary, ignore_index=ignore_index, multidim_average=multidim_average),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )


def _sk_stat_scores_multiclass(preds, target, ignore_index, multidim_average, average):
    preds = preds.numpy()
    target = target.numpy()
    if preds.ndim == target.ndim + 1:
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    confmat = sk_confusion_matrix(y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES)))
    tp = np.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    tn = confmat.sum() - (fp + fn + tp)

    res = np.stack([tp, fp, tn, fn, tp + fp + tn + fn], 1)
    if average == "micro":
        return res.sum(0)
    elif average == "macro":
        return res.mean(0)
    elif average == "weighted":
        w = tp + fn
        return res * (w / w.sum()).reshape(-1, 1)
    elif average is None or average == "none":
        return res


@pytest.mark.parametrize("input", _multiclass_cases)
@pytest.mark.parametrize("ignore_index", [None, 0, -1])
@pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
class TestMulticlassStatScores(MetricTester):
    def test_multiclass_stat_scores_functional(self, input, ignore_index, multidim_average, average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_stat_scores,
            sk_metric=partial(
                _sk_stat_scores_multiclass,
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


def _sk_stat_scores_multilabel(preds, target, ignore_index, multidim_average, average):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    stat_scores = []
    for i in range(preds.shape[1]):
        p, t = preds[:, i], target[:, i]
        if ignore_index is not None:
            idx = t == ignore_index
            t = t[~idx]
            p = p[~idx]
        tn, fp, fn, tp = sk_confusion_matrix(t, p, labels=[0, 1]).ravel()
        stat_scores.append(np.array([tp, fp, tn, fn, tp + fp + tn + fn]))
    return np.stack(stat_scores, axis=0)


@pytest.mark.parametrize("input", _multilabel_cases)
@pytest.mark.parametrize("ignore_index", [None, 0, -1])
@pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
@pytest.mark.parametrize("average", ["micro", "macro", "samples"])
class TestMultilabelyStatScores(MetricTester):
    def test_multilabel_stat_scores_functional(self, input, ignore_index, multidim_average, average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_stat_scores,
            sk_metric=partial(
                _sk_stat_scores_multilabel,
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


# -------------------------- Old stuff --------------------------

# def _sk_stat_scores(preds, target, reduce, num_classes, multiclass, ignore_index, top_k, threshold, mdmc_reduce=None):
#     # todo: `mdmc_reduce` is unused
#     preds, target, _ = _input_format_classification(
#         preds, target, threshold=threshold, num_classes=num_classes, multiclass=multiclass, top_k=top_k
#     )
#     sk_preds, sk_target = preds.numpy(), target.numpy()

#     if reduce != "macro" and ignore_index is not None and preds.shape[1] > 1:
#         sk_preds = np.delete(sk_preds, ignore_index, 1)
#         sk_target = np.delete(sk_target, ignore_index, 1)

#     if preds.shape[1] == 1 and reduce == "samples":
#         sk_target = sk_target.T
#         sk_preds = sk_preds.T

#     sk_stats = multilabel_confusion_matrix(
#         sk_target, sk_preds, samplewise=(reduce == "samples") and preds.shape[1] != 1
#     )

#     if preds.shape[1] == 1 and reduce != "samples":
#         sk_stats = sk_stats[[1]].reshape(-1, 4)[:, [3, 1, 0, 2]]
#     else:
#         sk_stats = sk_stats.reshape(-1, 4)[:, [3, 1, 0, 2]]

#     if reduce == "micro":
#         sk_stats = sk_stats.sum(axis=0, keepdims=True)

#     sk_stats = np.concatenate([sk_stats, sk_stats[:, [3]] + sk_stats[:, [0]]], 1)

#     if reduce == "micro":
#         sk_stats = sk_stats[0]

#     if reduce == "macro" and ignore_index is not None and preds.shape[1]:
#         sk_stats[ignore_index, :] = -1

#     return sk_stats


# def _sk_stat_scores_mdim_mcls(
#     preds, target, reduce, mdmc_reduce, num_classes, multiclass, ignore_index, top_k, threshold
# ):
#     preds, target, _ = _input_format_classification(
#         preds, target, threshold=threshold, num_classes=num_classes, multiclass=multiclass, top_k=top_k
#     )

#     if mdmc_reduce == "global":
#         preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
#         target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

#         return _sk_stat_scores(preds, target, reduce, None, False, ignore_index, top_k, threshold)
#     if mdmc_reduce == "samplewise":
#         scores = []

#         for i in range(preds.shape[0]):
#             pred_i = preds[i, ...].T
#             target_i = target[i, ...].T
#             scores_i = _sk_stat_scores(pred_i, target_i, reduce, None, False, ignore_index, top_k, threshold)

#             scores.append(np.expand_dims(scores_i, 0))

#         return np.concatenate(scores)


# @pytest.mark.parametrize(
#     "reduce, mdmc_reduce, num_classes, inputs, ignore_index",
#     [
#         ["unknown", None, None, _input_binary, None],
#         ["micro", "unknown", None, _input_binary, None],
#         ["macro", None, None, _input_binary, None],
#         ["micro", None, None, _input_mdmc_prob, None],
#         ["micro", None, None, _input_binary_prob, 0],
#         ["micro", None, None, _input_mcls_prob, NUM_CLASSES],
#         ["micro", None, NUM_CLASSES, _input_mcls_prob, NUM_CLASSES],
#     ],
# )
# def test_wrong_params(reduce, mdmc_reduce, num_classes, inputs, ignore_index):
#     """Test a combination of parameters that are invalid and should raise an error.

#     This includes invalid ``reduce`` and ``mdmc_reduce`` parameter values, not setting ``num_classes`` when
#     ``reduce='macro'`, not setting ``mdmc_reduce`` when inputs are multi-dim multi-class``, setting ``ignore_index``
#     when inputs are binary, as well as setting ``ignore_index`` to a value higher than the number of classes.
#     """
#     with pytest.raises(ValueError):
#         stat_scores(
#             inputs.preds[0], inputs.target[0], reduce, mdmc_reduce, num_classes=num_classes, ignore_index=ignore_index
#         )

#     with pytest.raises(ValueError):
#         sts = StatScores(reduce=reduce, mdmc_reduce=mdmc_reduce, num_classes=num_classes, ignore_index=ignore_index)
#         sts(inputs.preds[0], inputs.target[0])


# @pytest.mark.parametrize("ignore_index", [None, 0])
# @pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
# @pytest.mark.parametrize(
#     "preds, target, sk_fn, mdmc_reduce, num_classes, multiclass, top_k, threshold",
#     [
#         (_input_binary_logits.preds, _input_binary_logits.target, _sk_stat_scores, None, 1, None, None, 0.0),
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_stat_scores, None, 1, None, None, 0.5),
#         (_input_binary.preds, _input_binary.target, _sk_stat_scores, None, 1, False, None, 0.5),
#         (_input_mlb_logits.preds, _input_mlb_logits.target, _sk_stat_scores, None, NUM_CLASSES, None, None, 0.0),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, None, 0.5),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, 2, 0.5),
#         (_input_mcls.preds, _input_mcls.target, _sk_stat_scores, None, NUM_CLASSES, False, None, 0.5),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, None, 0.5),
#         (_input_mcls_logits.preds, _input_mcls_logits.target, _sk_stat_scores, None, NUM_CLASSES, None, None, 0.0),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, 2, 0.0),
#         (_input_multiclass.preds, _input_multiclass.target, _sk_stat_scores, None, NUM_CLASSES, None, None, 0.0),
#         (
#             _input_mdmc.preds,
#             _input_mdmc.target,
#             _sk_stat_scores_mdim_mcls,
#             "samplewise",
#             NUM_CLASSES,
#             None,
#             None,
#             0.0
#         ),
#         (
#             _input_mdmc_prob.preds,
#             _input_mdmc_prob.target,
#             _sk_stat_scores_mdim_mcls,
#             "samplewise",
#             NUM_CLASSES,
#             None,
#             None,
#             0.0,
#         ),
#         (_input_mdmc.preds, _input_mdmc.target, _sk_stat_scores_mdim_mcls, "global", NUM_CLASSES, None, None, 0.0),
#         (
#             _input_mdmc_prob.preds,
#             _input_mdmc_prob.target,
#             _sk_stat_scores_mdim_mcls,
#             "global",
#             NUM_CLASSES,
#             None,
#             None,
#             0.0,
#         ),
#     ],
# )
# class TestStatScores(MetricTester):
#     # DDP tests temporarily disabled due to hanging issues
#     @pytest.mark.parametrize("ddp", [False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     @pytest.mark.parametrize("dtype", [torch.float, torch.double])
#     def test_stat_scores_class(
#         self,
#         ddp: bool,
#         dist_sync_on_step: bool,
#         dtype: torch.dtype,
#         sk_fn: Callable,
#         preds: Tensor,
#         target: Tensor,
#         reduce: str,
#         mdmc_reduce: Optional[str],
#         num_classes: Optional[int],
#         multiclass: Optional[bool],
#         ignore_index: Optional[int],
#         top_k: Optional[int],
#         threshold: Optional[float],
#     ):
#         if ignore_index is not None and preds.ndim == 2:
#             pytest.skip("Skipping ignore_index test with binary inputs.")

#         if preds.is_floating_point():
#             preds = preds.to(dtype)
#         if target.is_floating_point():
#             target = target.to(dtype)

#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=StatScores,
#             sk_metric=partial(
#                 sk_fn,
#                 reduce=reduce,
#                 mdmc_reduce=mdmc_reduce,
#                 num_classes=num_classes,
#                 multiclass=multiclass,
#                 ignore_index=ignore_index,
#                 top_k=top_k,
#                 threshold=threshold,
#             ),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={
#                 "num_classes": num_classes,
#                 "reduce": reduce,
#                 "mdmc_reduce": mdmc_reduce,
#                 "threshold": threshold,
#                 "multiclass": multiclass,
#                 "ignore_index": ignore_index,
#                 "top_k": top_k,
#             },
#         )

#     def test_stat_scores_fn(
#         self,
#         sk_fn: Callable,
#         preds: Tensor,
#         target: Tensor,
#         reduce: str,
#         mdmc_reduce: Optional[str],
#         num_classes: Optional[int],
#         multiclass: Optional[bool],
#         ignore_index: Optional[int],
#         top_k: Optional[int],
#         threshold: Optional[float],
#     ):
#         if ignore_index is not None and preds.ndim == 2:
#             pytest.skip("Skipping ignore_index test with binary inputs.")

#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=stat_scores,
#             sk_metric=partial(
#                 sk_fn,
#                 reduce=reduce,
#                 mdmc_reduce=mdmc_reduce,
#                 num_classes=num_classes,
#                 multiclass=multiclass,
#                 ignore_index=ignore_index,
#                 top_k=top_k,
#                 threshold=threshold,
#             ),
#             metric_args={
#                 "num_classes": num_classes,
#                 "reduce": reduce,
#                 "mdmc_reduce": mdmc_reduce,
#                 "threshold": threshold,
#                 "multiclass": multiclass,
#                 "ignore_index": ignore_index,
#                 "top_k": top_k,
#             },
#         )

#     def test_stat_scores_differentiability(
#         self,
#         sk_fn: Callable,
#         preds: Tensor,
#         target: Tensor,
#         reduce: str,
#         mdmc_reduce: Optional[str],
#         num_classes: Optional[int],
#         multiclass: Optional[bool],
#         ignore_index: Optional[int],
#         top_k: Optional[int],
#         threshold: Optional[float],
#     ):
#         if ignore_index is not None and preds.ndim == 2:
#             pytest.skip("Skipping ignore_index test with binary inputs.")

#         self.run_differentiability_test(
#             preds,
#             target,
#             metric_module=StatScores,
#             metric_functional=stat_scores,
#             metric_args={
#                 "num_classes": num_classes,
#                 "reduce": reduce,
#                 "mdmc_reduce": mdmc_reduce,
#                 "threshold": threshold,
#                 "multiclass": multiclass,
#                 "ignore_index": ignore_index,
#                 "top_k": top_k,
#             },
#         )


# _mc_k_target = tensor([0, 1, 2])
# _mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
# _ml_k_target = tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
# _ml_k_preds = tensor([[0.9, 0.2, 0.75], [0.1, 0.7, 0.8], [0.6, 0.1, 0.7]])


# @pytest.mark.parametrize(
#     "k, preds, target, reduce, expected",
#     [
#         (1, _mc_k_preds, _mc_k_target, "micro", tensor([2, 1, 5, 1, 3])),
#         (2, _mc_k_preds, _mc_k_target, "micro", tensor([3, 3, 3, 0, 3])),
#         (1, _ml_k_preds, _ml_k_target, "micro", tensor([0, 3, 3, 3, 3])),
#         (2, _ml_k_preds, _ml_k_target, "micro", tensor([1, 5, 1, 2, 3])),
#         (1, _mc_k_preds, _mc_k_target, "macro", tensor([[0, 1, 1], [0, 1, 0], [2, 1, 2], [1, 0, 0], [1, 1, 1]])),
#         (2, _mc_k_preds, _mc_k_target, "macro", tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1]])),
#         (1, _ml_k_preds, _ml_k_target, "macro", tensor([[0, 0, 0], [1, 0, 2], [1, 1, 1], [1, 2, 0], [1, 2, 0]])),
#         (2, _ml_k_preds, _ml_k_target, "macro", tensor([[0, 1, 0], [2, 0, 3], [0, 1, 0], [1, 1, 0], [1, 2, 0]])),
#     ],
# )
# def test_top_k(k: int, preds: Tensor, target: Tensor, reduce: str, expected: Tensor):
#     """A simple test to check that top_k works as expected."""

#     class_metric = StatScores(top_k=k, reduce=reduce, num_classes=3)
#     class_metric.update(preds, target)

#     assert torch.equal(class_metric.compute(), expected.T)
#     assert torch.equal(stat_scores(preds, target, top_k=k, reduce=reduce, num_classes=3), expected.T)
