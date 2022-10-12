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
from netcal.metrics import ECE, MCE
from scipy.special import expit as sigmoid
from scipy.special import softmax

from torchmetrics.classification.calibration_error import BinaryCalibrationError, MulticlassCalibrationError
from torchmetrics.functional.classification.calibration_error import (
    binary_calibration_error,
    multiclass_calibration_error,
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_9
from unittests.classification.inputs import _binary_cases, _multiclass_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import NUM_CLASSES, MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _sk_binary_calibration_error(preds, target, n_bins, norm, ignore_index):
    preds = preds.numpy().flatten()
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    metric = ECE if norm == "l1" else MCE
    return metric(n_bins).measure(preds, target)


@pytest.mark.parametrize("input", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryCalibrationError(MetricTester):
    @pytest.mark.parametrize("n_bins", [10, 15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_calibration_error(self, input, ddp, n_bins, norm, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryCalibrationError,
            sk_metric=partial(_sk_binary_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
            metric_args={
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("n_bins", [10, 15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_calibration_error_functional(self, input, n_bins, norm, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_calibration_error,
            sk_metric=partial(_sk_binary_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
            metric_args={
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_calibration_error_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryCalibrationError,
            metric_functional=binary_calibration_error,
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_calibration_error_dtype_cpu(self, input, dtype):
        preds, target = input

        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryCalibrationError,
            metric_functional=binary_calibration_error,
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_calibration_error_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryCalibrationError,
            metric_functional=binary_calibration_error,
            dtype=dtype,
        )


def _sk_multiclass_calibration_error(preds, target, n_bins, norm, ignore_index):
    preds = preds.numpy()
    target = target.numpy().flatten()
    if not ((0 < preds) & (preds < 1)).all():
        preds = softmax(preds, 1)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target, preds = remove_ignore_index(target, preds, ignore_index)
    metric = ECE if norm == "l1" else MCE
    return metric(n_bins).measure(preds, target)


@pytest.mark.parametrize(
    "input", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassCalibrationError(MetricTester):
    @pytest.mark.parametrize("n_bins", [15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_calibration_error(self, input, ddp, n_bins, norm, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassCalibrationError,
            sk_metric=partial(_sk_multiclass_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("n_bins", [15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_calibration_error_functional(self, input, n_bins, norm, ignore_index):
        preds, target = input
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_calibration_error,
            sk_metric=partial(_sk_multiclass_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index),
            metric_args={
                "num_classes": NUM_CLASSES,
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_calibration_error_differentiability(self, input):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassCalibrationError,
            metric_functional=multiclass_calibration_error,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_calibration_error_dtype_cpu(self, input, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_9:
            pytest.xfail(reason="torch.max in metric not supported before pytorch v1.9 for cpu + half")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.softmax in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassCalibrationError,
            metric_functional=multiclass_calibration_error,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_calibration_error_dtype_gpu(self, input, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassCalibrationError,
            metric_functional=multiclass_calibration_error,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )
