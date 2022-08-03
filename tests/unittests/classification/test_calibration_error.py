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
import re

import numpy as np
import pytest
from scipy.special import softmax as _softmax

from torchmetrics.functional.classification.calibration_error import (
    binary_calibration_error,
    multiclass_calibration_error
)
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType
from unittests.classification.inputs import _binary_cases, _multiclass_cases
from unittests.helpers import seed_all

# TODO: replace this with official sklearn implementation after next sklearn release
from unittests.helpers.reference_metrics import _calibration_error as sk_calib
from unittests.helpers.testers import THRESHOLD, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_calibration(preds, target, n_bins, norm, debias=False, ignore_index=None):
    _, _, mode = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = preds.numpy(), target.numpy()
    if mode == DataType.BINARY:
        if not np.logical_and(0 <= sk_preds, sk_preds <= 1).all():
            sk_preds = 1.0 / (1 + np.exp(-sk_preds))  # sigmoid transform
    if mode == DataType.MULTICLASS:
        if not np.logical_and(0 <= sk_preds, sk_preds <= 1).all():
            sk_preds = _softmax(sk_preds, axis=1)
        # binary label is whether or not the predicted class is correct
        sk_target = np.equal(np.argmax(sk_preds, axis=1), sk_target)
        sk_preds = np.max(sk_preds, axis=1)
    elif mode == DataType.MULTIDIM_MULTICLASS:
        # reshape from shape (N, C, ...) to (N*EXTRA_DIMS, C)
        sk_preds = np.transpose(sk_preds, axes=(0, 2, 1))
        sk_preds = sk_preds.reshape(np.prod(sk_preds.shape[:-1]), sk_preds.shape[-1])
        # reshape from shape (N, ...) to (N*EXTRA_DIMS,)
        # binary label is whether or not the predicted class is correct
        sk_target = np.equal(np.argmax(sk_preds, axis=1), sk_target.flatten())
        sk_preds = np.max(sk_preds, axis=1)
    return sk_calib(y_true=sk_target, y_prob=sk_preds, norm=norm, n_bins=n_bins, reduce_bias=debias)



@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryCalibrationError(MetricTester):
    # @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    # @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    # @pytest.mark.parametrize("ddp", [True, False])
    # def test_binary_calibration_error(self, input, ddp, normalize, ignore_index):
    #     preds, target = input
    #     if ignore_index is not None:
    #         target = inject_ignore_index(target, ignore_index)
    #     self.run_class_metric_test(
    #         ddp=ddp,
    #         preds=preds,
    #         target=target,
    #         metric_class=BinaryCalibrationError,
    #         sk_metric=partial(_sk_calibration_error_binary, normalize=normalize, ignore_index=ignore_index),
    #         metric_args={
    #             "threshold": THRESHOLD,
    #             "normalize": normalize,
    #             "ignore_index": ignore_index,
    #         },
    #     )
    @pytest.mark.parametrize("n_bins", [10, 15, 20])
    @pytest.mark.parametrize("norm", ["l1", "l2", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_calibration_error_functional(self, input, n_bins, norm, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_calibration_error,
            sk_metric=partial(_sk_calibration, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
            metric_args={
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

#     def test_binary_calibration_error_differentiability(self, input):
#         preds, target = input
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=BinaryCalibrationError,
#             metric_functional=binary_calibration_error,
#             metric_args={"threshold": THRESHOLD},
#         )

#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_binary_calibration_error_dtype_cpu(self, input, dtype):
#         preds, target = input
#         if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
#             pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
#         if (preds < 0).any() and dtype == torch.half:
#             pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
#         self.run_precision_test_cpu(
#             preds=preds,
#             target=target,
#             metric_module=BinaryCalibrationError,
#             metric_functional=binary_calibration_error,
#             metric_args={"threshold": THRESHOLD},
#             dtype=dtype,
#         )

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_binary_calibration_error_dtype_gpu(self, input, dtype):
#         preds, target = input
#         self.run_precision_test_gpu(
#             preds=preds,
#             target=target,
#             metric_module=BinaryCalibrationError,
#             metric_functional=binary_calibration_error,
#             metric_args={"threshold": THRESHOLD},
#             dtype=dtype,
#         )


# def _sk_calibration_error_multiclass(preds, target, normalize=None, ignore_index=None):
#     preds = preds.numpy()
#     target = target.numpy()
#     if np.issubdtype(preds.dtype, np.floating):
#         preds = np.argmax(preds, axis=1)
#     preds = preds.flatten()
#     target = target.flatten()
#     target, preds = remove_ignore_index(target, preds, ignore_index)
#     return sk_calibration_error(y_true=target, y_pred=preds, normalize=normalize, labels=list(range(NUM_CLASSES)))


# @pytest.mark.parametrize("input", _multiclass_cases)
# class TestMulticlassCalibrationError(MetricTester):
#     @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
#     @pytest.mark.parametrize("ignore_index", [None, -1, 0])
#     @pytest.mark.parametrize("ddp", [True, False])
#     def test_multiclass_calibration_error(self, input, ddp, normalize, ignore_index):
#         preds, target = input
#         if ignore_index is not None:
#             target = inject_ignore_index(target, ignore_index)
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=MulticlassCalibrationError,
#             sk_metric=partial(_sk_calibration_error_multiclass, normalize=normalize, ignore_index=ignore_index),
#             metric_args={
#                 "num_classes": NUM_CLASSES,
#                 "normalize": normalize,
#                 "ignore_index": ignore_index,
#             },
#         )

#     @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
#     @pytest.mark.parametrize("ignore_index", [None, -1, 0])
#     def test_multiclass_calibration_error_functional(self, input, normalize, ignore_index):
#         preds, target = input
#         if ignore_index is not None:
#             target = inject_ignore_index(target, ignore_index)
#         self.run_functional_metric_test(
#             preds=preds,
#             target=target,
#             metric_functional=multiclass_calibration_error,
#             sk_metric=partial(_sk_calibration_error_multiclass, normalize=normalize, ignore_index=ignore_index),
#             metric_args={
#                 "num_classes": NUM_CLASSES,
#                 "normalize": normalize,
#                 "ignore_index": ignore_index,
#             },
#         )

#     def test_multiclass_calibration_error_differentiability(self, input):
#         preds, target = input
#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassCalibrationError,
#             metric_functional=multiclass_calibration_error,
#             metric_args={"num_classes": NUM_CLASSES},
#         )

#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multiclass_calibration_error_dtype_cpu(self, input, dtype):
#         preds, target = input
#         if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
#             pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
#         self.run_precision_test_cpu(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassCalibrationError,
#             metric_functional=multiclass_calibration_error,
#             metric_args={"num_classes": NUM_CLASSES},
#             dtype=dtype,
#         )

#     @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
#     @pytest.mark.parametrize("dtype", [torch.half, torch.double])
#     def test_multiclass_calibration_error_dtype_gpu(self, input, dtype):
#         preds, target = input
#         self.run_precision_test_gpu(
#             preds=preds,
#             target=target,
#             metric_module=MulticlassCalibrationError,
#             metric_functional=multiclass_calibration_error,
#             metric_args={"num_classes": NUM_CLASSES},
#             dtype=dtype,
#         )


# -------------------------- Old stuff --------------------------

# @pytest.mark.parametrize("n_bins", [10, 15, 20])
# @pytest.mark.parametrize("norm", ["l1", "l2", "max"])
# @pytest.mark.parametrize(
#     "preds, target",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target),
#         (_input_binary_logits.preds, _input_binary_logits.target),
#         (_input_mcls_prob.preds, _input_mcls_prob.target),
#         (_input_mcls_logits.preds, _input_mcls_logits.target),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target),
#     ],
# )
# class TestCE(MetricTester):
#     @pytest.mark.parametrize("ddp", [True, False])
#     @pytest.mark.parametrize("dist_sync_on_step", [True, False])
#     def test_ce(self, preds, target, n_bins, ddp, dist_sync_on_step, norm):
#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=CalibrationError,
#             sk_metric=functools.partial(_sk_calibration, n_bins=n_bins, norm=norm),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={"n_bins": n_bins, "norm": norm},
#         )

#     def test_ce_functional(self, preds, target, n_bins, norm):
#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=calibration_error,
#             sk_metric=functools.partial(_sk_calibration, n_bins=n_bins, norm=norm),
#             metric_args={"n_bins": n_bins, "norm": norm},
#         )


# @pytest.mark.parametrize("preds, targets", [(_input_mlb_prob.preds, _input_mlb_prob.target)])
# def test_invalid_input(preds, targets):
#     for p, t in zip(preds, targets):
#         with pytest.raises(
#             ValueError,
#             match=re.escape(
#                 f"Calibration error is not well-defined for data with size {p.size()} and targets {t.size()}."
#             ),
#         ):
#             calibration_error(p, t)


# @pytest.mark.parametrize(
#     "preds, target",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target),
#         (_input_mcls_prob.preds, _input_mcls_prob.target),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target),
#     ],
# )
# def test_invalid_norm(preds, target):
#     with pytest.raises(ValueError, match="Norm l3 is not supported. Please select from l1, l2, or max. "):
#         calibration_error(preds, target, norm="l3")


# @pytest.mark.parametrize("n_bins", [-10, -1, "fsd"])
# @pytest.mark.parametrize(
#     "preds, targets",
#     [
#         (_input_binary_prob.preds, _input_binary_prob.target),
#         (_input_mcls_prob.preds, _input_mcls_prob.target),
#         (_input_mdmc_prob.preds, _input_mdmc_prob.target),
#     ],
# )
# def test_invalid_bins(preds, targets, n_bins):
#     for p, t in zip(preds, targets):
#         with pytest.raises(ValueError, match=f"Expected argument `n_bins` to be a int larger than 0 but got {n_bins}"):
#             calibration_error(p, t, n_bins=n_bins)
