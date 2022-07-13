from functools import partial

import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa

from torchmetrics.classification.cohen_kappa import BinaryCohenKappa, CohenKappa, MulticlassCohenKappa
from torchmetrics.functional.classification.cohen_kappa import binary_cohen_kappa, cohen_kappa, multiclass_cohen_kappa
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester, inject_ignore_index

seed_all(42)


def _sk_cohen_kappa_binary(preds, target, weights=None, ignore_index=None):
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
    return sk_cohen_kappa(y1=target, y2=preds, weights=weights)


@pytest.mark.parametrize("input", _binary_cases)
class TestBinaryConfusionMatrix(MetricTester):
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
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
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

    if ignore_index is not None:
        idx = target == ignore_index
        target = target[~idx]
        preds = preds[~idx]
    return sk_cohen_kappa(y1=target, y2=preds, weights=weights)


# -------------------------- Old stuff --------------------------

# def _sk_cohen_kappa_binary_prob(preds, target, weights=None):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_binary(preds, target, weights=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_multilabel_prob(preds, target, weights=None):
#     sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_multilabel(preds, target, weights=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_multiclass_prob(preds, target, weights=None):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_multiclass(preds, target, weights=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_multidim_multiclass_prob(preds, target, weights=None):
#     sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# def _sk_cohen_kappa_multidim_multiclass(preds, target, weights=None):
#     sk_preds = preds.view(-1).numpy()
#     sk_target = target.view(-1).numpy()

#     return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


# @pytest.mark.parametrize("weights", ["linear", "quadratic", None])
# @pytest.mark.parametrize(
#     "preds, target, sk_metric, num_classes",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target, _sk_cohen_kappa_binary_prob, 2),
#         (_input_binary.preds, _input_binary.target, _sk_cohen_kappa_binary, 2),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_cohen_kappa_multilabel_prob, 2),
#         (_input_mlb.preds, _input_mlb.target, _sk_cohen_kappa_multilabel, 2),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_cohen_kappa_multiclass_prob, NUM_CLASSES),
#         (_input_mcls.preds, _input_mcls.target, _sk_cohen_kappa_multiclass, NUM_CLASSES),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_cohen_kappa_multidim_multiclass_prob, NUM_CLASSES),
#         (_input_mdmc.preds, _input_mdmc.target, _sk_cohen_kappa_multidim_multiclass, NUM_CLASSES),
#     ],
# )
# class TestCohenKappa(MetricTester):
#     atol = 1e-5

#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_cohen_kappa(self, weights, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=CohenKappa,
#             sk_metric=partial(sk_metric, weights=weights),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "weights": weights},
#         )

#     def test_cohen_kappa_functional(self, weights, preds, target, sk_metric, num_classes):
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=cohen_kappa,
#             sk_metric=partial(sk_metric, weights=weights),
#             metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "weights": weights},
#         )

#     def test_cohen_kappa_differentiability(self, preds, target, sk_metric, weights, num_classes):
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=CohenKappa,
#             metric_functional=cohen_kappa,
#             metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "weights": weights},
#         )


# def test_warning_on_wrong_weights(tmpdir):
#     preds = torch.randint(3, size=(20,))
#     target = torch.randint(3, size=(20,))

#     with pytest.raises(ValueError, match=".* ``weights`` but should be either None, 'linear' or 'quadratic'"):
#         cohen_kappa(preds, target, num_classes=3, weights="unknown_arg")
