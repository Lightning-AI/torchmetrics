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
import pytest
from sklearn.metrics import hamming_loss as sk_hamming_loss
import numpy as np
import torch
from functools import partial
from torchmetrics.functional.classification.hamming import binary_hamming_distance
from unittests.helpers import seed_all
from torchmetrics.classification.hamming import BinaryHammingDistance
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers.testers import THRESHOLD, MetricTester, inject_ignore_index
from scipy.special import expit as sigmoid
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)


def _sk_hamming_loss(target, preds):
    score = sk_hamming_loss(target, preds)
    return score if not np.isnan(score) else 1.0


def _sk_hamming_distance_binary(preds, target, ignore_index, multidim_average):
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
        return _sk_hamming_loss(target, preds)
    else:
        res = []
        for pred, true in zip(preds, target):
            pred = pred.flatten()
            true = true.flatten()
            if ignore_index is not None:
                idx = true == ignore_index
                true = true[~idx]
                pred = pred[~idx]
            res.append(_sk_hamming_loss(true, pred))
        return np.stack(res)


@pytest.mark.parametrize("input", _binary_cases)
class TestBinaryHammingDistance(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ddp", [False, True])
    def test_binary_hamming_distance(self, ddp, input, ignore_index, multidim_average):
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
            metric_class=BinaryHammingDistance,
            sk_metric=partial(
                _sk_hamming_distance_binary, ignore_index=ignore_index, multidim_average=multidim_average
            ),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "multidim_average": multidim_average},
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_hamming_distance_functional(self, input, ignore_index, multidim_average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_hamming_distance,
            sk_metric=partial(
                _sk_hamming_distance_binary, ignore_index=ignore_index, multidim_average=multidim_average
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_hamming_distance_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryHammingDistance,
            metric_functional=binary_hamming_distance,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_hamming_distance_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryHammingDistance,
            metric_functional=binary_hamming_distance,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_hamming_distance_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryHammingDistance,
            metric_functional=binary_hamming_distance,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _sk_hamming_distance_multiclass(preds, target, ignore_index, multidim_average, average):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)
    if multidim_average == "global":
        preds = preds.numpy().flatten()
        target = target.numpy().flatten()

        if ignore_index is not None:
            idx = target == ignore_index
            target = target[~idx]
            preds = preds[~idx]
        confmat = sk_confusion_matrix(y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES)))
        tp = np.diag(confmat)
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)

        if average == "micro":
            return _calc_hamming_distance(tn.sum(), fp.sum())

        res = _calc_hamming_distance(tn, fp)
        if average == "macro":
            return res.mean(0)
        elif average == "weighted":
            w = tp + fn
            return (res * (w / w.sum()).reshape(-1, 1)).sum(0)
        elif average is None or average == "none":
            return res
    else:
        preds = preds.numpy()
        target = target.numpy()

        res = []
        for pred, true in zip(preds, target):
            pred = pred.flatten()
            true = true.flatten()

            if ignore_index is not None:
                idx = true == ignore_index
                true = true[~idx]
                pred = pred[~idx]
            confmat = sk_confusion_matrix(y_true=true, y_pred=pred, labels=list(range(NUM_CLASSES)))
            tp = np.diag(confmat)
            fp = confmat.sum(0) - tp
            fn = confmat.sum(1) - tp
            tn = confmat.sum() - (fp + fn + tp)
            if average == "micro":
                res.append(_calc_hamming_distance(tn.sum(), fp.sum()))

            r = _calc_hamming_distance(tn, fp)
            if average == "macro":
                res.append(r.mean(0))
            elif average == "weighted":
                w = tp + fn
                res.append((r * (w / w.sum()).reshape(-1, 1)).sum(0))
            elif average is None or average == "none":
                res.append(r)
        return np.stack(res, 0)


# @pytest.mark.parametrize("input", _multiclass_cases)
# class TestMulticlassHammingDistance(MetricTester):
#     @pytest.mark.parametrize("ignore_index", [None, 0, -1])
#     @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
#     @pytest.mark.parametrize("average", ["micro", "macro", None])
#     @pytest.mark.parametrize("ddp", [True, False])
#     def test_multiclass_hamming_distance(self, ddp, input, ignore_index, multidim_average, average):
#         preds, target = input
#         if ignore_index == -1:
#             target = inject_ignore_index(target, ignore_index)
#         if multidim_average == "samplewise" and target.ndim < 3:
#             pytest.skip("samplewise and non-multidim arrays are not valid")
#         if multidim_average == "samplewise" and ddp:
#             pytest.skip("samplewise and ddp give different order than non ddp")

#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=MulticlassHammingDistance,
#             sk_metric=partial(
#                 _sk_hamming_distance_multiclass,
#                 ignore_index=ignore_index,
#                 multidim_average=multidim_average,
#                 average=average,
#             ),
#             metric_args={
#                 "ignore_index": ignore_index,
#                 "multidim_average": multidim_average,
#                 "average": average,
#                 "num_classes": NUM_CLASSES,
#             },
#         )

#     @pytest.mark.parametrize("ignore_index", [None, 0, -1])
#     @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
#     @pytest.mark.parametrize("average", ["micro", "macro", None])
#     def test_multiclass_hamming_distance_functional(self, input, ignore_index, multidim_average, average):
#         preds, target = input
#         if ignore_index == -1:
#             target = inject_ignore_index(target, ignore_index)
#         if multidim_average == "samplewise" and target.ndim < 3:
#             pytest.skip("samplewise and non-multidim arrays are not valid")

#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=multiclass_hamming_distance,
#             sk_metric=partial(
#                 _sk_hamming_distance_multiclass,
#                 ignore_index=ignore_index,
#                 multidim_average=multidim_average,
#                 average=average,
#             ),
#             metric_args={
#                 "ignore_index": ignore_index,
#                 "multidim_average": multidim_average,
#                 "average": average,
#                 "num_classes": NUM_CLASSES,
#             },
#         )

#     def test_multiclass_hamming_distance_differentiability(self, input):
#         preds, target = input
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassHammingDistance,
#             metric_functional=multiclass_hamming_distance,
#             metric_args={"num_classes": NUM_CLASSES},
#         )

#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multiclass_hamming_distance_dtype_cpu(self, input, dtype):
#         preds, target = input
#         if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
#             pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
#         if (preds < 0).any() and dtype == torch.half:
#             pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
#         self.run_precision_test_cpu(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassHammingDistance,
#             metric_functional=multiclass_hamming_distance,
#             metric_args={"num_classes": NUM_CLASSES},
#             dtype=dtype,
#         )

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multiclass_hamming_distance_dtype_gpu(self, input, dtype):
#         preds, target = input
#         self.run_precision_test_gpu(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassHammingDistance,
#             metric_functional=multiclass_hamming_distance,
#             metric_args={"num_classes": NUM_CLASSES},
#             dtype=dtype,
#         )


# _mc_k_target = tensor([0, 1, 2])
# _mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])


# @pytest.mark.parametrize(
#     "k, preds, target, average, expected_spec",
#     [
#         (1, _mc_k_preds, _mc_k_target, "micro", tensor(5 / 6)),
#         (2, _mc_k_preds, _mc_k_target, "micro", tensor(1 / 2)),
#     ],
# )
# def test_top_k(k: int, preds: Tensor, target: Tensor, average: str, expected_spec: Tensor):
#     """A simple test to check that top_k works as expected."""
#     class_metric = MulticlassHammingDistance(top_k=k, average=average, num_classes=3)
#     class_metric.update(preds, target)

#     assert torch.equal(class_metric.compute(), expected_spec)
#     assert torch.equal(
#         multiclass_hamming_distance(preds, target, top_k=k, average=average, num_classes=3), expected_spec
#     )


# def _sk_hamming_distance_multilabel(preds, target, ignore_index, multidim_average, average):
#     preds = preds.numpy()
#     target = target.numpy()
#     if np.issubdtype(preds.dtype, np.floating):
#         if not ((0 < preds) & (preds < 1)).all():
#             preds = sigmoid(preds)
#         preds = (preds >= THRESHOLD).astype(np.uint8)
#     preds = preds.reshape(*preds.shape[:2], -1)
#     target = target.reshape(*target.shape[:2], -1)
#     if multidim_average == "global":
#         tns, fps = [], []
#         hamming_distance = []
#         for i in range(preds.shape[1]):
#             p, t = preds[:, i].flatten(), target[:, i].flatten()
#             if ignore_index is not None:
#                 idx = t == ignore_index
#                 t = t[~idx]
#                 p = p[~idx]
#             tn, fp, fn, tp = sk_confusion_matrix(t, p, labels=[0, 1]).ravel()
#             tns.append(tn)
#             fps.append(fp)

#         tn = np.array(tns)
#         fp = np.array(fps)
#         if average == "micro":
#             return _calc_hamming_distance(tn.sum(), fp.sum())

#         res = _calc_hamming_distance(tn, fp)
#         if average == "macro":
#             return res.mean(0)
#         elif average == "weighted":
#             w = res[:, 0] + res[:, 3]
#             return (res * (w / w.sum()).reshape(-1, 1)).sum(0)
#         elif average is None or average == "none":
#             return res
#     else:
#         hamming_distance = []
#         for i in range(preds.shape[0]):
#             tns, fps = [], []
#             for j in range(preds.shape[1]):
#                 pred, true = preds[i, j], target[i, j]
#                 if ignore_index is not None:
#                     idx = true == ignore_index
#                     true = true[~idx]
#                     pred = pred[~idx]
#                 tn, fp, _, _ = sk_confusion_matrix(true, pred, labels=[0, 1]).ravel()
#                 tns.append(tn)
#                 fps.append(fp)
#             tn = np.array(tns)
#             fp = np.array(fps)
#             if average == "micro":
#                 hamming_distance.append(_calc_hamming_distance(tn.sum(), fp.sum()))
#             else:
#                 hamming_distance.append(_calc_hamming_distance(tn, fp))

#         res = np.stack(hamming_distance, 0)
#         if average == "micro" or average is None or average == "none":
#             return res
#         elif average == "macro":
#             return res.mean(-1)
#         elif average == "weighted":
#             w = res[:, 0, :] + res[:, 3, :]
#             return (res * (w / w.sum())[:, np.newaxis]).sum(-1)
#         elif average is None or average == "none":
#             return np.moveaxis(res, 1, -1)


# @pytest.mark.parametrize("input", _multilabel_cases)
# class TestMultilabelHammingDistance(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("ignore_index", [None, 0, -1])
#     @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
#     @pytest.mark.parametrize("average", ["micro", "macro", None])
#     def test_multilabel_hamming_distance(self, ddp, input, ignore_index, multidim_average, average):
#         preds, target = input
#         if ignore_index == -1:
#             target = inject_ignore_index(target, ignore_index)
#         if multidim_average == "samplewise" and preds.ndim < 4:
#             pytest.skip("samplewise and non-multidim arrays are not valid")
#         if multidim_average == "samplewise" and ddp:
#             pytest.skip("samplewise and ddp give different order than non ddp")

#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=MultilabelHammingDistance,
#             sk_metric=partial(
#                 _sk_hamming_distance_multilabel,
#                 ignore_index=ignore_index,
#                 multidim_average=multidim_average,
#                 average=average,
#             ),
#             metric_args={
#                 "num_labels": NUM_CLASSES,
#                 "threshold": THRESHOLD,
#                 "ignore_index": ignore_index,
#                 "multidim_average": multidim_average,
#                 "average": average,
#             },
#         )

#     @pytest.mark.parametrize("ignore_index", [None, 0, -1])
#     @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
#     @pytest.mark.parametrize("average", ["micro", "macro", None])
#     def test_multilabel_hamming_distance_functional(self, input, ignore_index, multidim_average, average):
#         preds, target = input
#         if ignore_index == -1:
#             target = inject_ignore_index(target, ignore_index)
#         if multidim_average == "samplewise" and preds.ndim < 4:
#             pytest.skip("samplewise and non-multidim arrays are not valid")

#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=multilabel_hamming_distance,
#             sk_metric=partial(
#                 _sk_hamming_distance_multilabel,
#                 ignore_index=ignore_index,
#                 multidim_average=multidim_average,
#                 average=average,
#             ),
#             metric_args={
#                 "num_labels": NUM_CLASSES,
#                 "threshold": THRESHOLD,
#                 "ignore_index": ignore_index,
#                 "multidim_average": multidim_average,
#                 "average": average,
#             },
#         )

#     def test_multilabel_hamming_distance_differentiability(self, input):
#         preds, target = input
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=MultilabelHammingDistance,
#             metric_functional=multilabel_hamming_distance,
#             metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
#         )

#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multilabel_hamming_distance_dtype_cpu(self, input, dtype):
#         preds, target = input
#         if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
#             pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
#         if (preds < 0).any() and dtype == torch.half:
#             pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
#         self.run_precision_test_cpu(
#             preds=preds,
#             target=target,
#             metric_module=MultilabelHammingDistance,
#             metric_functional=multilabel_hamming_distance,
#             metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
#             dtype=dtype,
#         )

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multilabel_hamming_distance_dtype_gpu(self, input, dtype):
#         preds, target = input
#         self.run_precision_test_gpu(
#             preds=preds,
#             target=target,
#             metric_module=MultilabelHammingDistance,
#             metric_functional=multilabel_hamming_distance,
#             metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
#             dtype=dtype,
#         )


# -------------------------- Old stuff --------------------------

# def _sk_hamming_loss(preds, target):
#     sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD)
#     sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
#     sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

#     return sk_hamming_loss(y_true=sk_target, y_pred=sk_preds)


# @pytest.mark.parametrize(
#     "preds, target",
#     [
#         (_input_binary_logits.preds, _input_binary_logits.target),
#         (_input_binary_prob.preds, _input_binary_prob.target),
#         (_input_binary.preds, _input_binary.target),
#         (_input_mlb_logits.preds, _input_mlb_logits.target),
#         (_input_mlb_prob.preds, _input_mlb_prob.target),
#         (_input_mlb.preds, _input_mlb.target),
#         (_input_mcls_logits.preds, _input_mcls_logits.target),
#         (_input_mcls_prob.preds, _input_mcls_prob.target),
#         (_input_mcls.preds, _input_mcls.target),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target),
#         (_input_mdmc.preds, _input_mdmc.target),
#         (_input_mlmd_prob.preds, _input_mlmd_prob.target),
#         (_input_mlmd.preds, _input_mlmd.target),
#     ],
# )
# class TestHammingDistance(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [False, True])
#     def test_hamming_distance_class(self, ddp, dist_sync_on_step, preds, target):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=HammingDistance,
#             sk_metric=_sk_hamming_loss,
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"threshold": THRESHOLD},
#         )

#     def test_hamming_distance_fn(self, preds, target):
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=hamming_distance,
#             sk_metric=_sk_hamming_loss,
#             metric_args={"threshold": THRESHOLD},
#         )

#     def test_hamming_distance_differentiability(self, preds, target):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=HammingDistance,
#             metric_functional=hamming_distance,
#             metric_args={"threshold": THRESHOLD},
#         )


# @pytest.mark.parametrize("threshold", [1.5])
# def test_wrong_params(threshold):
#     preds, target = _input_mcls_prob.preds, _input_mcls_prob.target

#     with pytest.raises(ValueError):
#         ham_dist = HammingDistance(threshold=threshold)
#         ham_dist(preds, target)
#         ham_dist.compute()

#     with pytest.raises(ValueError):
#         hamming_distance(preds, target, threshold=threshold)
