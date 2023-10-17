# Copyright The Lightning team.
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
from torchmetrics.classification.calibration_error import (
    BinaryCalibrationError,
    CalibrationError,
    MulticlassCalibrationError,
)
from torchmetrics.functional.classification.calibration_error import (
    binary_calibration_error,
    multiclass_calibration_error,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_13

from unittests import NUM_CLASSES
from unittests.classification.inputs import _binary_cases, _multiclass_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index

seed_all(42)


def _netcal_binary_calibration_error(preds, target, n_bins, norm, ignore_index):
    preds = preds.numpy().flatten()
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(target, preds, ignore_index)
    metric = ECE if norm == "l1" else MCE
    return metric(n_bins).measure(preds, target)


@pytest.mark.parametrize("inputs", (_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]))
class TestBinaryCalibrationError(MetricTester):
    """Test class for `BinaryCalibrationError` metric."""

    @pytest.mark.parametrize("n_bins", [10, 15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_calibration_error(self, inputs, ddp, n_bins, norm, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryCalibrationError,
            reference_metric=partial(
                _netcal_binary_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index
            ),
            metric_args={
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("n_bins", [10, 15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_calibration_error_functional(self, inputs, n_bins, norm, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_calibration_error,
            reference_metric=partial(
                _netcal_binary_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index
            ),
            metric_args={
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_calibration_error_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryCalibrationError,
            metric_functional=binary_calibration_error,
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_calibration_error_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_13:
            pytest.xfail(reason="torch.linspace in metric not supported before pytorch v1.13 for cpu + half")
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
    def test_binary_calibration_error_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_13:
            pytest.xfail(reason="torch.searchsorted in metric not supported before pytorch v1.13 for gpu + half")
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryCalibrationError,
            metric_functional=binary_calibration_error,
            dtype=dtype,
        )


def test_binary_with_zero_pred():
    """Test that metric works with edge case where confidence is zero for a bin."""
    preds = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    target = torch.tensor([0, 0, 1, 1, 1])
    assert binary_calibration_error(preds, target, n_bins=2, norm="l1") == torch.tensor(0.6)


def _netcal_multiclass_calibration_error(preds, target, n_bins, norm, ignore_index):
    preds = preds.numpy()
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target, preds = remove_ignore_index(target, preds, ignore_index)
    metric = ECE if norm == "l1" else MCE
    return metric(n_bins).measure(preds, target)


@pytest.mark.parametrize(
    "inputs", (_multiclass_cases[1], _multiclass_cases[2], _multiclass_cases[4], _multiclass_cases[5])
)
class TestMulticlassCalibrationError(MetricTester):
    """Test class for `MulticlassCalibrationError` metric."""

    @pytest.mark.parametrize("n_bins", [15, 20])
    @pytest.mark.parametrize("norm", ["l1", "max"])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_calibration_error(self, inputs, ddp, n_bins, norm, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassCalibrationError,
            reference_metric=partial(
                _netcal_multiclass_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index
            ),
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
    def test_multiclass_calibration_error_functional(self, inputs, n_bins, norm, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_calibration_error,
            reference_metric=partial(
                _netcal_multiclass_calibration_error, n_bins=n_bins, norm=norm, ignore_index=ignore_index
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_calibration_error_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassCalibrationError,
            metric_functional=multiclass_calibration_error,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_calibration_error_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
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
    def test_multiclass_calibration_error_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassCalibrationError,
            metric_functional=multiclass_calibration_error,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


def test_corner_case_due_to_dtype():
    """Test that metric works with edge case where the precision is really important for the right result.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/1907

    """
    preds = torch.tensor(
        [0.9000, 0.9000, 0.9000, 0.9000, 0.9000, 0.8000, 0.8000, 0.0100, 0.3300, 0.3400, 0.9900, 0.6100],
        dtype=torch.float64,
    )
    target = torch.tensor([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0])

    assert np.allclose(
        ECE(99).measure(preds.numpy(), target.numpy()), binary_calibration_error(preds, target, n_bins=99)
    ), "The metric should be close to the netcal implementation"
    assert np.allclose(
        ECE(100).measure(preds.numpy(), target.numpy()), binary_calibration_error(preds, target, n_bins=100)
    ), "The metric should be close to the netcal implementation"


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryCalibrationError, {"task": "binary"}),
        (MulticlassCalibrationError, {"task": "multiclass", "num_classes": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=CalibrationError):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match=r"Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
